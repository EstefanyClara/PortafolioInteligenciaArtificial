import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

input_file = 'cardiac-training.csv'

#Le asignamos los nombres de las columnas. 
col_names= ['Edad', 'Estado_civil', 'Sexo', 'Categoria_Peso', 'Colesterol', 'Manejo_stress', 'Trat_ansiedad', '2do_Ataque_Corazon' ]
df= pd.read_csv(input_file, header=0, names= col_names)


print(df.head())
X = df.loc[:, df.columns != '2do_Ataque_Corazon']
y = df['2do_Ataque_Corazon'].values



#Dividir el conjunto de datos en 2 partes: entrenamiento y prueba

train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=0, shuffle=True)

