import argparse
import os
import torch
import yaml
from lightning_model import LitFlipyFlopy
from spline_model import LitBSplineNN
import utils_unispec


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def main():
    parser = argparse.ArgumentParser(
        description="Export Altimeter models to TorchScript and ONNX"
    )
    parser.add_argument(
        "altimeter_outpath", help="Path to save the TorchScript Altimeter model"
    )
    parser.add_argument(
        "spline_outpath", help="Path to save the spline ONNX model"
    )
    parser.add_argument(
        "--dic-config", required=True, help="Path to dictionary config YAML"
    )
    parser.add_argument(
        "--model-config", required=True, help="Path to model config YAML"
    )
    parser.add_argument(
        "--model-ckpt", required=True, help="Path to model checkpoint"
    )
    args = parser.parse_args()
    
    # Instantiate DicObj
    with open(args.dic_config, "r") as stream:
        dconfig = yaml.safe_load(stream)
    with open(
        os.path.join(os.path.dirname(__file__), "../config/mods.yaml"), "r"
    ) as stream:
        mod_config = yaml.safe_load(stream)

    D = utils_unispec.DicObj(
        dconfig["ion_dictionary_path"],
        mod_config,
        dconfig["seq_len"],
        dconfig["chlim"],
    )
    L = utils_unispec.LoadObj(D, embed=True)

    # Instantiate model
    with open(args.model_config) as stream:
        model_config = yaml.safe_load(stream)
    config = {
        "dic_config": args.dic_config,
        "model_config": args.model_config,
        "model_ckpt": args.model_ckpt,
    }
    model = LitFlipyFlopy.load_from_checkpoint(
        args.model_ckpt, config=config, model_config=model_config
    )


    input_seq = torch.zeros((1, L.channels, D.seq_len), dtype=torch.float32, device=device)
    input_ch = torch.zeros((1,1), dtype=torch.float32, device=device)
    
    input_sample = [input_seq, input_ch]
    input_names = ["inp", "inpch"]
    output_names = ["coefficients", "knots", "AUCs"]
    
    print(model.model.get_knots())
    


    script = torch.jit.trace(
        lambda seq, ch: model.forward_coef([seq, ch]),
        (input_seq, input_ch)             
    )
    torch.jit.save(script, args.altimeter_outpath)
    
    
    # repeat for splines
    model2 = LitBSplineNN()
    input_coef = torch.zeros((1, 4, 380), dtype=torch.float32, device=device)
    input_knots = model.model.get_knots().unsqueeze(0).to(device)
    input_ce = torch.zeros((1,1), dtype=torch.float32, device=device)
    input_sample = (input_coef, input_knots, input_ce)
    y = model2(*input_sample)
    print(y.shape)
    
    input_names = ["coefficients", "knots", "inpce"]
    output_names = ["intensities"]
    
    model2.to_onnx(
        args.spline_outpath,
        input_sample,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "coefficients": {0: "batch_size"},
            "knots": {0: "batch_size"},
            "inpce": {0: "batch_size"},
            "intensities": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()