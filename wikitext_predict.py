import hydra
from omegaconf import DictConfig, OmegaConf
from wikitext_transformer import TransformerModel
from pathlib import Path
import torch


@hydra.main(config_path="config", config_name="wikitext_predict_config")
def run(cfg: DictConfig):

    path = cfg.path
    

    model = TransformerModel.load_from_checkpoint(cfg.checkpoint)
    data = torch.cat([torch.tensor([model.vocab[token]
                                    for token in model.tokenizer(cfg.text)])])
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

        '''
        for j in range(argmax.shape[1]) :
            index = argmax[i][j]
            print('max_arg', index)
            print('value', model.vocab.itos[index])
        '''

    print('cfg.text', cfg.text)
    print('vocab', text_result)


if __name__ == "__main__":
    run()
