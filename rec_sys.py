import pandas as pd 
import scipy.sparse as sparse
import implicit
import numpy as np

eslite_db = pd.read_csv('/Users/lucaschang/Desktop/誠品生活/eslite_ec_data.csv')

print(eslite_db.info())

eslite_db[['member_id']].sort_values(by=['member_id'])

eslite_db.sub_order_item_id.unique().shape

csr_matrix = sparse.csr_matrix(eslite_db[['member_id','product_id']].values)

csr_matrix.toarray().shape

#the member * product table
df1 = (eslite_db.head(100).assign(new = 1)
         .drop_duplicates(subset=['product_id','member_id'])
         .pivot('product_id','member_id','new')
         .fillna(0)
         .astype(int))

print(df1.head(10))

print(df1.info)

df1_csr = sparse.csr_matrix(df1)

mem_prod_df1_csr = sparse.csr_matrix(mem_prod_df1)

model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)

alpha = 15
data = (df1_csr * alpha).astype('double')

# Fit the model
model.fit(data)

product_id = 0
n_similar = 10

#？？model.user_factors要怎麼查的出來
person_vecs = model.user_factors
content_vecs = model.item_factors

#?
content_norms = np.sqrt((content_vecs * content_vecs).sum(axis=1))

#?好像是做標準化？
scores = content_vecs.dot(content_vecs[product_id]) / content_norms

#?
top_idx = np.argpartition(scores, -n_similar)[-n_similar:]

#?zip 是幹嘛的，similar是相似度
similar = sorted(zip(top_idx, scores[top_idx] / content_norms[product_id]), key=lambda x: -x[1])
