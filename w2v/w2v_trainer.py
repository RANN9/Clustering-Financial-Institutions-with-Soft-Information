# Copyright 2024 RAN
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from w2v_model_and_trainer_utility import *
import wandb
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# purpose-specific w2v SGNS main trainer function
def main():
    seed = 9
    device = torch.device("cuda")  # cpu or cuda
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    data_config, vocab_list, _, word_to_idx_dict, idx_to_word_dict, input_list, output_list, negative_samples = load_resources_sgns()
    firm_benchm, currencies_benchm, countries_benchm = load_benchmark_lists()

    model_config = {
        "global_seed": seed,
        "model": "w2v_sgns",
        "device": str(device),
        "vocab_size": len(vocab_list),
        "embedding_dim": 300,
        "optimizer": "Adam",
        "learning_rate": 1e-04,
        "epochs": 3,
        "batch_size": 512,
    }

    dataset = TensorDataset(
    torch.tensor(input_list, dtype=torch.int32),
    torch.tensor(output_list, dtype=torch.int32),
    torch.tensor(negative_samples, dtype=torch.int32),
    )
    dataloader = DataLoader(
        dataset, batch_size=model_config["batch_size"], shuffle=True, pin_memory=True, num_workers=4
    )

    clean_up(input_list, output_list, negative_samples)

    model = SkipGramNS(model_config["vocab_size"], model_config["embedding_dim"]).to(device)
    OptimizerClass = getattr(optim, model_config["optimizer"])
    optimizer = OptimizerClass(model.parameters(), model_config["learning_rate"])

    data_and_model_config = {**data_config, **model_config}
    with open(Path("../../embedding_data/w2v/data_and_model_config.json"), "w") as f:
        json.dump(data_and_model_config, f, indent=4)
    wandb.init(project="w2v_sgns", config=data_and_model_config)

    for epoch in range(model_config["epochs"]):
        epoch_loss = 0
        for i, (input, output, neg_output) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            loss = model.forward(input_pos= input.to(device, non_blocking=True),
                                output_pos = output.to(device, non_blocking=True),
                                output_neg = neg_output.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            wandb.log({"current_mini_batch_loss": loss.item()})

            if i % 50000 == 0:
                print(f"Epoch {epoch+1}, Mini-batch {i+1}, Current Mini-batch Loss: {loss.item()}")
                temp_embeddings = model.output_embedding.weight.data.cpu()
                fin_sim_score = similar_to_many_evaluation_metrics(
                    temp_embeddings,
                    word_to_idx_dict,
                    idx_to_word_dict,
                    inputs=firm_benchm,
                    top_n=10,
                )
                currency_sim_score = similar_to_many_evaluation_metrics(
                    temp_embeddings,
                    word_to_idx_dict,
                    idx_to_word_dict,
                    inputs=currencies_benchm,
                    top_n=10,
                )
                country_sim_score = similar_to_many_evaluation_metrics(
                    temp_embeddings,
                    word_to_idx_dict,
                    idx_to_word_dict,
                    inputs=countries_benchm,
                    top_n=10,
                )
                wandb.log(
                    {
                        "fin_sim_score": fin_sim_score,
                        "currency_sim_score": currency_sim_score,
                        "country_sim_score": country_sim_score,
                    }
                )
        wandb.log({"epoch_loss": epoch_loss / i + 1})
        torch.save(model.output_embedding.weight.data.cpu(), Path("../../embedding_data/w2v")/(f"w2v_epoch_{epoch+1}_output_embedding.pth"))
        epoch_performance = {
            "epoch": epoch + 1,
            "epoch_loss": epoch_loss / i + 1,
            "fin_sim_score": fin_sim_score,
            "currency_sim_score": currency_sim_score,
            "country_sim_score": country_sim_score,
        }
        with open(Path(f"../../embedding_data/w2v/epoch_{epoch+1}_performance.json"), "w") as f:
            json.dump(epoch_performance, f, indent=4)
    wandb.finish()

if __name__ == '__main__':
    main()