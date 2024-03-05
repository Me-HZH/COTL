import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, input_size)

        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def build_classifier():
    return RandomForestClassifier()


folder_path = './data'

project_list = os.listdir(folder_path)

col = ['nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp', 'ns', 'nd']

df_result = pd.DataFrame()

for source_project in project_list:
    for target_project in project_list:
        if source_project == target_project:
            continue
        print(source_project + " -> " + target_project)
        source_data = pd.read_excel(f'./datasets/{source_project}', index_col=False).dropna(axis=0, how='any')
        target_data = pd.read_excel(f'./datasets/{target_project}', index_col=False).dropna(axis=0, how='any')

        y_src = source_data['contains_bug']
        X_src = source_data[col]
        y_tgt = target_data['contains_bug']
        X_tgt = target_data[col]

        scaler = StandardScaler()
        X_src = scaler.fit_transform(X_src)
        X_tgt = scaler.transform(X_tgt)

        input_size = len(col)
        hidden_size = 16
        batch_size = 32
        num_epochs = 100
        learning_rate = 0.001

        src_dataset = TensorDataset(torch.tensor(X_src, dtype=torch.float32), torch.tensor(y_src, dtype=torch.float32))
        src_dataloader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True)

        tgt_dataset = TensorDataset(torch.tensor(X_tgt, dtype=torch.float32), torch.tensor(y_tgt, dtype=torch.float32))
        tgt_dataloader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True)

        generator = Generator(input_size, hidden_size)
        discriminator = Discriminator(input_size, hidden_size)

        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for src_data, tgt_data in zip(src_dataloader, tgt_dataloader):
                src_inputs, src_labels = src_data
                tgt_inputs, tgt_labels = tgt_data

                optimizer_d.zero_grad()
                src_preds = discriminator(src_inputs)
                tgt_preds = discriminator(tgt_inputs)

                tgt_labels = torch.zeros_like(tgt_preds)

                loss_d = criterion(src_preds.squeeze(), src_labels) + criterion(tgt_preds, tgt_labels)
                loss_d.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                src_gen_features = generator(src_inputs)
                tgt_gen_features = generator(tgt_inputs)

                src_disc_preds = discriminator(src_gen_features)
                tgt_disc_preds = discriminator(tgt_gen_features)

                loss_g = criterion(src_disc_preds.squeeze(), src_labels) \
                         + criterion(tgt_disc_preds,torch.ones_like(tgt_disc_preds))
                loss_g.backward()
                optimizer_g.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")


        with torch.no_grad():
            src_common_features = generator(torch.tensor(X_src, dtype=torch.float32))
            tgt_common_features = generator(torch.tensor(X_tgt, dtype=torch.float32))

        classifier = build_classifier()
        classifier.fit(src_common_features, y_src)
        test_label = classifier.predict(tgt_common_features)

        acc = accuracy_score(y_true=test_label, y_pred=y_tgt)
        f1 = f1_score(y_true=test_label, y_pred=y_tgt)
        precision = precision_score(y_true=test_label, y_pred=y_tgt)
        recall = recall_score(y_true=test_label, y_pred=y_tgt)
        g_mean = (recall * (1 - precision)) ** 0.5

        result = {
            'source': source_project,
            'target': target_project,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'g_mean': g_mean
        }

        df_result = df_result.append(result, ignore_index=True)

df_result.to_csv('KAL_results.csv', mode='a', header=False, index=False)
