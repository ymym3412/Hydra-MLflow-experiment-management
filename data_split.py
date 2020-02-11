import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


titles, articles, labels = [], [], []
news_list = ['dokujo-tsushin', 'it-life-hack', 'kaden-channel', 'livedoor-homme', 'movie-enter', 'peachy', 'smax', 'sports-watch', 'topic-news']
for i, media in enumerate(news_list):
    files = os.listdir(Path('data', 'text', media))
    for file_name in files:
        if file_name == 'LICENSE.txt':
            continue
        with Path('data', 'text', media, file_name).open(encoding='utf-8') as f:
            lines = [line for line in f]
            title = lines[2].replace('\n', '')
            text = ''.join(lines[3:])
            titles.append(title)
            articles.append(text.replace('\n', ''))
            labels.append(i)


df = pd.DataFrame({'title': titles, 'article': articles, 'label': labels})

train_X, test_X, train_y, test_y = train_test_split(df[['article', 'title']], df['label'], stratify=df['label'], test_size=0.3, random_state=0)
val_X, test_X, val_y, test_y = train_test_split(test_X[['article', 'title']], test_y, stratify=test_y, test_size=0.5, random_state=0)

train_X.to_csv('data/train.csv', index=False)
val_X.to_csv('data/val.csv', index=False)
test_X.to_csv('data/test.csv', index=False)

train_y.to_csv('data/train_label.csv', index=False)
val_y.to_csv('data/val_label.csv', index=False)
test_y.to_csv('data/test_label.csv', index=False)
