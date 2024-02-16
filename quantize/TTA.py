import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def tta(
    lm,
    args,
    dataloader,
    fp_lm,
    logger=None
):
    logger.info("Start Test time adaptation ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False

    fp_model = fp_lm.model
    fp_model.config.use_cache = False

    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

        fp_layers = fp_model.model.layers

        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        print(layers)
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (len(dataloader), lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input 
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= len(dataloader):
                break
            try:
                model(batch.to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")
    torch.cuda.empty_cache()


    # same input of first layer for fp model and quant model
    quant_inps = copy.deepcopy(inps)
    fp_inps = copy.deepcopy(inps)

    attention_mask = cache["attention_mask"]
    attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    loss_func = torch.nn.MSELoss()
    # loss_func = torch.nn.L1Loss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    cossim = nn.CosineSimilarity(dim=2)
    
    
    for i in range(len(layers)):
        if i == (len(layers)-1): #(len(layers)-1)
            logger.info(f"=== Start adapt layer {i} ===")

            fp_layer = fp_layers[i].to(dev)
            qfp_layer = DecoderLayer(fp_lm.model.config, fp_layer, args)
            qfp_layer = qfp_layer.to(dev)          
                                    
            # init smooth parameters

            use_shift = True 

            qfp_layer.set_quant_state(weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
            qfp_layer.let = args.let

            qfp_layer.set_quant_state(weight_quant=False, act_quant=False)
            if args.epochs > 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        for j in range(len(dataloader)):
                            fp_inps[j] = qfp_layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            
            if args.let:
                # init channel-wise scaling and shift
                qfp_layer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(fp_layer.self_attn.q_proj.out_features,device=dev, dtype=dtype),requires_grad = True))
                for name,module in qfp_layer.named_modules():
                    if isinstance(module, QuantLinear):
                        for key in pairs.keys():
                            if key in name:
                                weight = module.weight.max(dim=0)[0]
                                tta_scale = (torch.ones_like(weight)) #weight
                                # shift = torch.zeros_like(scale)
                                tta_shift = torch.zeros_like(weight)
                                qfp_layer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(tta_shift,requires_grad = True))
                                qfp_layer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(tta_scale,requires_grad = True))

            if args.epochs > 0:
                with torch.no_grad():
                    qfp_layer.float()      # required for AMP training
                # create optimizer
                optimizer = torch.optim.AdamW(
                    [{"params":qfp_layer.let_parameters(use_shift),"lr":args.let_lr}, {"params":qfp_layer.lwc_parameters(),"lr":1e-2}],weight_decay=args.wd) # 1e-3,1e-2; best: 1e-5,1e-4
                loss_scaler = utils.NativeScalerWithGradNormCount()

                for epochs in range(5):
                    loss_list = []
                    norm_list = []
                    for j in range(len(dataloader)//args.batch_size):    
                        index = j * args.batch_size
                        # obtain output of quantization model
                        with traincast():
                            qfp_layer.smooth_and_quant_temporary() # 
                            quant_out = qfp_layer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            # quant_out  False
                            loss = loss_func((fp_inps[index:index+args.batch_size,]), quant_out) #tmp_fp_tta

                            cos = cossim(quant_out,fp_inps[index:index+args.batch_size,]).mean().abs()
                            loss -= torch.log(cos)

                        if not math.isfinite(loss.item()):
                            logger.info("Loss is NAN, stopping adapting")
                            # quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            pdb.set_trace()
                            
                        loss_list.append(loss.data)
                        optimizer.zero_grad()
                        norm = loss_scaler(loss, optimizer,parameters=qfp_layer.rlq_parameters(use_shift))
                        norm_list.append(norm.data)

                    loss_mean = torch.stack(loss_list).mean()
                    norm_mean = torch.stack(norm_list).mean()
                    logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                qfp_layer.clear_temp_variable()
                del optimizer
        
            # real smooth and quantization
            qfp_layer.smooth_and_quant_inplace()       
            if args.epochs>0:
                qfp_layer.register_scales_and_zeros()
                qfp_layer.half()
                layers[i] = qfp_layer.to("cpu")
            else:
                qfp_layer.register_scales_and_zeros()
                qfp_layer.half()
                layers[i] = qfp_layer.to("cpu")
            del fp_layer
            torch.cuda.empty_cache()

        else:
            continue

    del inps
    del quant_inps
    del fp_inps
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model