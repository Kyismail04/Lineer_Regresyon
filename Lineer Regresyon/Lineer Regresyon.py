import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv("student.csv")
#print(df.head(3))
y=df["Marks"]
x=df.drop("Marks",axis=1)

l=LinearRegression()
model=l.fit(x,y)
print("tahmin: ",model.predict([[3,2]]))

print("skor: ",model.score(x,y))