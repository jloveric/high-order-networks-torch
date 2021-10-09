import hydra
from omegaconf import DictConfig, OmegaConf
from wikitext_transformer import TransformerModel
from pathlib import Path
import torch
import os

@hydra.main(config_path="config", config_name="wikitext_predict_config")
def run(cfg: DictConfig):

    path = cfg.path
    checkpoint_path = Path(cfg.path) / "checkpoints"
    checkpoint_list = os.listdir(checkpoint_path)
    print('checkpoint_list', checkpoint_list)
    checkpoint = checkpoint_path / checkpoint_list[0]

    model = TransformerModel.load_from_checkpoint(checkpoint)
    data = torch.unsqueeze(torch.cat([torch.tensor([model.vocab[token]
                                    for token in model.tokenizer(cfg.text)])]),dim=0)
    #data = torch.cat([torch.tensor([model.vocab[token]
    #                                for token in model.tokenizer(cfg.text)])])
    print('data.shape', data.shape)
    model.eval()

    print('data', data)
    print(model.tokenizer(cfg.text))
    print('model', model)
    src_mask = model.generate_square_subsequent_mask(data.shape[0])
    print('src_mask', src_mask)
    result = model(src=data, src_mask=src_mask)
    print('data', data)
    print('result.shape', result.shape)
    index_list = []

    argmax = torch.argmax(result, dim=2)
    text_result = []
    for i in range(argmax.shape[0]):
        text_result.append([model.vocab.itos[index] for index in argmax[i]])

    print('cfg.text', cfg.text)
    print('vocab', text_result)


if __name__ == "__main__":
    run()
