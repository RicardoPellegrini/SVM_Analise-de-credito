
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Importando database
base = pd.read_csv('credit_data.csv')


# In[3]:


# Estatísticas do database
base.describe()


# In[4]:


# Amostra dos dados
base.head()


# In[5]:


# Verificando dados com idade negativa
base.loc[base['age'] < 0]


# In[6]:


### Maneiras de contornar o problema das idades menores que zero

## 1) Apagar a coluna por inteiro (não recomendada, neste caso)
# base.drop('age', 1, inplace=True)

## 2) Apagar apenas os registros, por completo, que possuem essa incoerência
# base.drop(base[base.age < 0].index, inplace=True)

## 3) Preencher os valores com a média da coluna, apenas dos valores maiores que zero
media = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = media


# In[7]:


# Verificando valores nulos
base.loc[pd.isnull(base['age'])]


# In[8]:


# Divisão do dataset entre variáveis preditoras e target
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


# In[9]:


# Substituindo os valores missing pela média de cada coluna
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(previsores[:, 0:3])

previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])


# In[10]:


## Fazendo o escalonamento (normalização) dos atributos
from sklearn.preprocessing import StandardScaler

# Padronização
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Normalização
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# previsores = scaler.fit_transform(previsores)


# In[11]:


# Dividindo os dados em treino e teste
from sklearn.model_selection import train_test_split


# In[12]:


previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25, random_state=0)


# In[13]:


# Modelo SVC
from sklearn.svm import SVC


# In[32]:


classificador = SVC(kernel='rbf', random_state=1, C=2.0)


# In[33]:


classificador.fit(previsores_train, classe_train)


# In[34]:


# Testando o modelo criado à partir dos dados de treinamento
previsoes = classificador.predict(previsores_test)


# In[35]:


# Calculando a precisão do nosso modelo
from sklearn.metrics import confusion_matrix, accuracy_score


# In[36]:


precisao = accuracy_score(classe_test, previsoes)
precisao


# In[37]:


matriz = confusion_matrix(classe_test, previsoes)
matriz


# ## Resultado
# ### SVM (Classifier)
# 0.946 - kernel='linear', C=1.0  
# 0.968 - kernel='poly', C=1.0  
# 0.838 - kernel='sigmoid', C=1.0  
# 0.980 - kernel='rbf', C=1.0  
# 0.988 - kernel='rbf', C=2.0
# 
