import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


#Leer el dataset

input_file = 'sample.csv'
df = pd.read_csv(input_file, header=0)
print(df.values)

#Graficar los datos

colors= ("red", "blue")
plt.scatter(df['x'], df['y'], s= 300, c=df['label']), 
cmap= matplotlib.colors.ListedColormap(colors)
plt.show()


#Parte 5 obtener a partir del dataset los datos y las clases 

X= df[['x','y']].values
Y= df['label'].values


# Parte 2 entrenamiento y testing

train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=0, shuffle=True)
lda = LinearDiscriminantAnalysis()
lda = lda.fit(train_X, train_y)

# Parte 3: evaluación del modelo

y_pred = lda.predict(test_X)
print("Predicted vs Expected")
print(y_pred)
print(test_y)


print(classification_report(test_y, y_pred, digits=3))


# Parte 4: Matriz de confusión
print(confusion_matrix(test_y, y_pred))

#Realizar los mismos procedimiento utilizando regresión logística


print("Regresión Logística")
lr= LogisticRegression()
lr= lr.fit(train_X, train_y)

y_pred = lr.predict(test_X)
print("Predicted vs Expected")
print(y_pred)
print(test_y)



