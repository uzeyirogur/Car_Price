import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""%matplotlib inline

import warnings 
warnings.simplefilter(action="ignore",category=Warning)"""

data = pd.read_csv("data/Automobile.csv")
print(data.info())

#Kaç adet nümerik sütun olduğu
print(len(data.describe().columns))

#DATA PREPROCESSİNG
#her bir kolon içinde tekil sayıyı çıkartıcaz şimdi mesela fultype sayısı 2 ya gas ya diesel 
for col in data.columns :
    print(col,data[col].nunique())
    

#Kategorik olan sütünlar için değerler
for col in data.columns : 
    values = []
    
    if col not in data.describe().columns :
        for val in data[col].unique():
            values.append(val)
            
        print("{0} -> {1}".format(col,values))
        
#araba adlarını düzenleme sadece audi bmw alıcaz detayı almıcaz
print(data.CarName)
manufacturer = data["CarName"].apply(lambda x:x.split(" "))      #boşlukla ayırıp sadece ilk satırını aldık
manufacturer = data["CarName"].apply(lambda x:x.split(" ")[0])

#orjinal datayı değiştirmeyelim copy yapalım
df = data.copy()

#CarName çıkar
df.drop(columns=["CarName"],axis=1,inplace=True)
#Sadece üreticileri ekleyelim şimdi
df.insert(3,"manufacturer",manufacturer)
#Bazı sıkıntılar var o yüzden groupby yapıp bi görelim 
aded = df.groupby(by="manufacturer").count()  #incelediğimizde nissan Nissan var mazda maxda var bunların düzeletilmesi gerek

#hepsini küçük harf
df.manufacturer = df.manufacturer.str.lower()
#hatalı markaları düzeltelim
df.replace({
    "maxda":"mazda",
    "porcshe":"porsche",
    "toyouta":"toyota",
    "vokswagen":"vw",
    "volkswagen":"vw"},inplace = True)


#Tekli (Univariate) 
sns.countplot(x=df["symboling"])
plt.show()

fig = plt.figure(figsize=(20,12))

plt.subplot(2,3,1)
plt.title("Fueltype")
sns.countplot(x=df["fueltype"])
#benzinli(gas) çoğunlukta

plt.subplot(2,3,2)
plt.title("Fuelsystem")
sns.countplot(x=df["fuelsystem"])
#mpfi(multi point fuel injection) çoğunlukta,yeni nesil teknoloji

plt.subplot(2,3,3)
plt.title("Aspiration")
sns.countplot(x=df["aspiration"])
#çoğunluk standart beslemeli

plt.subplot(2,3,4)
plt.title("Door Number")
sns.countplot(x=df["doornumber"])
#çoğunluk 4 kapılı

plt.subplot(2,3,5)
plt.title("Car Body")
sns.countplot(x=df["carbody"])
#çoğunluk sedan

plt.subplot(2,3,6)
plt.title("Drive Wheel")
sns.countplot(x=df["drivewheel"])
#çekiş sistemi,standard çeker çooğunlukta 

plt.show()


#ikili(binary) bakma
#üretici dizel ve gas ile satış fiyatı çizimi   
plt.figure(figsize=(25,15))
plt.title("Üretici Fiyatları",fontsize=16)
sns.barplot(x=df.manufacturer,y=df.price,hue=df.fueltype,palette="Set2")   #palette renk
plt.xticks(rotation=90) #yazıları 90 derece döndürdük
plt.tight_layout()      #grafik düzeni için
plt.show()

#fiyat karşılaştırmaları 
plt.title("Engine Location")  #motor önde mi arkada mı
sns.countplot(x=df["enginelocation"])
sns.boxplot(x=df.enginelocation,y=df.price)
plt.show()
#burada çok büyük fark var arkada olanların fiyatıyla önde olanların fiyatında arkada olanlar çok daha yüksek


#pair plat ikili grafiklere bakıp çok önemsizleri eleme yapıcaz
print(df.columns)

#bazılarını alıyoruz
cols = ['wheelbase','carlength', 'carwidth', 'carheight', 'curbweight',
       'enginesize', 'boreratio', 'stroke',
       'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']

plt.figure(figsize=(20,12))

for i in range(len(cols)) :
    plt.subplot(5,3,i+1)
    plt.title(cols[i]+" - Fiyat")
    sns.regplot(x=df[cols[i]],y=df.price)

plt.tight_layout()
plt.show()

#gereksiz olanları çıkartıyoruz
data_new = df[['car_ID', 'symboling', 'fueltype', 'manufacturer', 'aspiration',
       'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
       'carlength', 'carwidth', 'curbweight', 'enginetype',
       'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio',
       'horsepower','price']]
print(len(data_new))

""" biraz zorluyor çalışırken en son yorumdan çıkart
#tüm sütunları ikili basma
sns.pairplot(data_new)
plt.show()
"""

#FEATURE ENGİNEERİNG
#datayı daha iyi modellemek için yeni özellikler(feature) bulma veya özelliklerden bazılarını eleme işlemi
#Tork(Toque)
#aracın çekiş gücüyle alakalı torku yüksek olanın fiyatı daha pahalı olması beklenir
#torque = 5252*HP/RPM
torque = df.horsepower*5252/df.peakrpm
data_new.insert(10,"torque",pd.Series(df.horsepower*5252/df.peakrpm,index=df.index))

#price-torque ilişkisi
plt.title("TORQUE-PRİCES",fontsize=16)
sns.regplot(x=data_new.torque,y=data_new.price)
plt.show()

#ORT YAKIT TÜKTETİMİ
#arabanın şehir içi ve şehir dışı otomatik yakıt tüketimi
data_new.insert(loc=10, column="fueleconomy", value=(0.55*data.citympg)+(0.45*data.highwaympg))

#price-fueleconomy ilişkisi
plt.title("FUELECONOMY-PRİCES",fontsize=16)
sns.regplot(x=data_new.fueleconomy,y=data_new.price)
plt.show()

data_new.drop(columns=["car_ID","manufacturer","doornumber","symboling","fuelsystem"],axis=1,inplace=True)


#Datayı copy
cars = data_new.copy()

#Kategorik verileri numerice dönüştürme
dummies_list = ["fueltype","aspiration","carbody","drivewheel","enginelocation","enginetype","cylindernumber",]

for i in dummies_list :
    temp_df = pd.get_dummies(eval("cars"+"."+i),drop_first=True)
    cars = pd.concat([cars,temp_df],axis=1)
    cars.drop([i],axis=1,inplace=True)
#drop_first ilk kolonu siler mesela std ve turbo var sadece turboyu alıyor zaten turbo sıfır ise std dir demek 

#TRAİN TEST
from sklearn.model_selection import train_test_split
x_train,x_test = train_test_split(cars,test_size=0.30,random_state=42)

#Scale etme
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
scale_cols= ["wheelbase","torque","carlength","carwidth","curbweight","enginesize","horsepower","    ","boreratio"]
x_train[scale_cols] = sc.fit_transform(x_train[scale_cols]) 

#y alma
y_train = x_train.pop("price") #oradan çıkart y_train ekle pop 


#ÇOKLU LİNEER REGRESYON İÇİN KÜTÜPHANE İMPORTLARI
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

lr = LinearRegression()
lr.fit(x_train,y_train)
#RFE (Recursive Feature Elimination)
#Backward Elimination yaparak girdilerimizi azaltmıştık RFE kendisi yapıyor bunu
#n_features_to_select sayı veriyoruz ona göre yapıyor
#Her elemede en önemsiz olanı eler

rfe = RFE(estimator=lr, n_features_to_select=10)
rfe = rfe.fit(x_train,y_train)
rfe_sonuc = rfe.support_
rfs_secilenler = list(zip(x_train.columns,rfe.support_,rfe.ranking_))

#rfe nin seçtikleri üzerinden yeni bir x_train oluşturma
x_train_rfe = x_train[x_train.columns[rfe.support_]]

#OLS ANALİZi
x_trainrfe_model = x_train_rfe.copy()
x_trainrfe_model = sm.add_constant(x_trainrfe_model)

ols = sm.OLS(y_train,x_trainrfe_model).fit()
print(ols.summary())

#rotor önemsiz 0.191 çok büyük çıkar
x_trainrfe_model = x_trainrfe_model.drop(["rotor"],axis=1)

def trainols(x,y) :
    x = sm.add_constant(x)
    ols = sm.OLS(y,x).fit()
    print(ols.summary())

trainols(x_trainrfe_model,y_train)
x_trainrfe_model = x_trainrfe_model.drop(["dohcv"],axis=1)
trainols(x_trainrfe_model,y_train)
x_trainrfe_model = x_trainrfe_model.drop(["five"],axis=1)
trainols(x_trainrfe_model,y_train)

#En son elde kalanlar 
x_train_final = x_train[["curbweight","enginesize","horsepower","rear","four","six","twelve"]]

lr_final = LinearRegression()
lr_final.fit(x_train_final, y_train)


katsayilar = pd.DataFrame(lr_final.coef_,index = ["curbweight","enginesize","horsepower","rear","four","six","twelve"],columns=["Katsayı"])
katsayilar = katsayilar.sort_values(by=["Katsayı"],ascending=False)







