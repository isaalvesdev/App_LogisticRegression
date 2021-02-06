import pandas as pd
import numpy as np 
import os
import statsmodels.api as sm
import scipy.stats as stats
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import logit
import streamlit as st 
import sklearn.metrics as metrics


def load_df(path,**kwargs):
    
    with open(path,'r',**kwargs) as file:
        
        result = pd.read_csv(file, sep='\s+')

    return result

def data_transform(df):
    
    df_temp1 = df.rename(columns = {'4':'ID','1':'LOW','28':'AGE',
                     '120':'LWT','3':'RACE',
                     '1.1':'SMOKE','1.2':'PTL','0':'HT','1.3':'UI',
                     '0.1':'FTV','709':'BWT'})
    
    dummies_race = pd.get_dummies(df_temp1["RACE"],prefix = ['RACE'],drop_first = True)
    
    df_temp2 = pd.concat([df_temp1,dummies_race],axis = 1)
    
    
    
    df_temp3 = df_temp2.drop(["RACE"],axis = 1)
   
    real_df = df_temp3.rename(columns = {"['RACE']_2":"RACE_2","['RACE']_3":"RACE_3"})
     
    return real_df


def odds_ratio(df,column1,column2):

    
    table = df.groupby("LOW").sum()[[column1,column2]].values
    odds,p_value = stats.fisher_exact(table)
    
    return {"Odds Ratio": round(odds,2), "P-Valor": round(p_value,2)}
    
    
def stepwise_selection(X, y,initial_list=[],pvalue_in=0.15,pvalue_out = 0.20,verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < pvalue_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            
            changed=True
            
            if verbose:
                
                print(f'Adicionando {best_feature} com p-valor {best_pval}')

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > pvalue_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                
                print(f'Removendo {worst_feature} with p-value {worst_pval}')
        if not changed:
            break
    return included
    

def model(Y,*args):
    
    dependente = Y
    independentes = sm.add_constant(*args)
    
    logit_mod = sm.Logit(dependente,independentes)
    result = logit_mod.fit(method = 'newton')
    return result            
            
def wald_test(list_coef,list_se):
    
    values = []
    for i,j in zip(list_coef,list_se):
        test = round(abs(i / j),2)
        values.append(test)
        pvalor = 1 - stats.norm.cdf(values,0,1)
        
    for index, pvalues in enumerate(pvalor):
        
        if pvalues < 0.05:
            
            
            print (f'Rejeitamos H0 E Concluimos queo coeficiente Beta {index+1} , com W = {values[index]} tem significância, com um p-valor de {round(pvalues,6)}.\n')
            
        elif pvalues > 0.05:
            
            print(f'Não Rejeitamos H0 E Concluimos que o coeficiente Beta {index+1}, com W = {values[index]} não significância, com um p-valor de {round(pvalues,6)}.\n')
                                      
                                       
def hosmer_lemeshow_test(df,groups):
    
    data_st = df.sort_values('predict')
    data_st['dcl'] = pd.qcut(data_st['predict'],groups)
    
    ys = data_st['LOW'].groupby(data_st.dcl).sum()
    yt = data['LOW'].groupby(data_st.dcl).count()
    yn = yt - ys
    
    yps = data_st['predict'].groupby(data_st.dcl).sum()
    ypt = data_st['predict'].groupby(data_st.dcl).count()
    ypn = ypt - yps
    
    hltest = ( ((ys - yps)**2 / yps) + ((yn - ypn)**2 / ypn) ).sum()
    pvalor = 1 - stats.chi2.cdf(hltest,groups-2)
    
    df = groups - 2
    print(f'O teste de Hosmer-Lemeshow: {hltest}, com p-valor: {pvalor} com {df} graus de liberdade')
    
def plot_probability(df):
        
    sns.regplot(x = df['LWT'].values,y = df['LOW'],
                    logistic=True, color = 'blue')
    plt.xlabel('Hand Weight in the Last Menstrual Period')
    plt.ylabel('Probability')
    plt.title('Logistic Regression:Probability of a baby being born with low weight')
    plt.grid()
    plt.show()
    
def confusion_matrix_plot(model):
        
    sns.heatmap(model.pred_table())
    plt.title("Confusion Matrix")    
    plt.show()
        
        
def plot_logit(model,df):
        
    variables = pd.DataFrame(df,columns = ['LWT', 'SMOKE', 'PTL', 'HT', 'UI','RACE_2','RACE_3'])  
    X = sm.add_constant(variables)
    predict = model.predict(X)
    logit_g = logit(predict)
    
    sns.regplot(x = variables['LWT'].values,y = logit_g,color = '0.1')    
    plt.xlabel('Hand Weight in the Last Menstrual Period')
    plt.ylabel('Logit')
    plt.title('Logit of a baby being born with low weight')
    plt.grid()
    plt.show()   
        
def residual_pearson(model,df):
    
    observed = []
    
    m = np.asarray(df.groupby(['LWT','LOW'])['LOW'].count().values)
    yi = df.groupby(['LWT','LOW']).groups.keys()
    predict = model.predict()
    r = np.ones((len(m)))

    
    for i,j in yi:
        observed.append(j)
    
    obs_2 = np.asarray(observed)
    
    for k in range(len(m)):
        
        r[k] = (obs_2[k] - (m[k] * predict[k]))/math.sqrt(m[k] * predict[k] * (1 - predict[k]))
    
    Chi_2 = np.round(np.sum(r**2),2)
    
    chi_tab = stats.chi2.ppf(1 - .05,df = len(m) - model.df_model + 1)
    p_valor = 1 - stats.chi2.cdf(Chi_2,len(m) - model.df_model + 1)

    print(f'Como a estatística qui-quadrado {Chi_2}, valor tabelado {round(chi_tab,2)} e o seu p-valor é {round(p_valor,7)}')
    
def roc_curve(model,df):
    
    variables = pd.DataFrame(df,columns = ['LWT', 'SMOKE', 'PTL', 'HT', 'UI','RACE_2','RACE_3'])  
    X = sm.add_constant(variables)
    probs = model.predict(X)
    fpr,tpr,threshold = metrics.roc_curve(y,probs)    
    roc_auc = metrics.auc(fpr,tpr)

    
    
    plt.title("Curva Roc")
    plt.plot(fpr,tpr,'b',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    


st.title('Identificando fatores de riscos associados ao nascimento de bebês com baixo peso.')
st.text('O objetivo deste projeto é identificar os fatores de riscos que levam um bebê a nascer com baixo peso (menos que 2500 gramas). Este modelo foi construido a partir do modelo de regressão logistica. \n'
        'Nesse estudo foram coletados dados de 189 mulheres, em que n1 = 59 tiveram bebês que nasceram com baixo peso e n0 = 130, mulheres que tiveram bebês com peso normal.\n\n'
        
        'Os dados foram coletados no Hospital “Baystate Medical Center” de Springfield, Massachusetts, Estados Unidos, no ano de 1986. No modelo construído sete variáveis foram consideradas importantes, \n'
        'como peso da mãe (LWT), raça (RACE), se fumou durante a gravidez (SMOKE), se tem histórico de trabalho de parto prematuro (PTL), se tem hipertensão (HT), se a mâe tem irritabilidade uterina (UI). \n'
        'Das variaveis mencionadas, apenas as variáveis trabalho de parto e hipertensão não foram significativas, porém, de acordo com referências pesquisadas, elas são importantes. A variável resposta considerada foi a variável LOW, \n'
        'baixo peso do bebê.')


base_path = os.path.dirname(os.path.abspath(__file__))

#Importando dados
df = load_df(os.path.join(base_path,"lowbwt.dat"))

    

#Dados Transformados
data = data_transform(df)

st.subheader('Dados:')
st.write(data.head())
    
#Verificando se há valores ausentes
#data.info()
#data.isnull().sum()
    
# Teste Odds Ratio com Raça 1 e Raça 2
odds_ratio(data,"RACE_2","RACE_3") 
    
#-------------------------------



X = pd.DataFrame(data, columns=['AGE', 'LWT', 'SMOKE', 'PTL', 'HT', 'UI', 'FTV', 'RACE_2', 'RACE_3'])
y = data["LOW"].values

stepwise_selection(X,y)
    
#Modelo com as Váriaveis selecionadas:
dependent_variable = data["LOW"].values
selected_variables = pd.DataFrame(data,columns = ['LWT', 'SMOKE', 'PTL', 'HT', 'UI','RACE_2','RACE_3'])
    
modelo = model(dependent_variable,selected_variables)

#Teste Hosmer_Lemeshow: 
data["predict"] = modelo.predict()
hosmer_lemeshow_test(data,10)

list_coef = []
    
for index,params in enumerate(modelo.params):
        
    if index > 0:
        list_coef.append(params)
       
list_se = np.sqrt([modelo.cov_params()['LWT']['LWT'],
            modelo.cov_params()['SMOKE']['SMOKE'],
            modelo.cov_params()['PTL']['PTL'],
            modelo.cov_params()['HT']['HT'],
            modelo.cov_params()['UI']['UI'],
            modelo.cov_params()['RACE_2']['RACE_2'],
            modelo.cov_params()['RACE_3']['RACE_3'] 
            ])    
    
    
#Teste de Wald
wald_test(list_coef,list_se)
    
#Probaility
st.subheader('Modelo Encontrado: \n')
st.latex(r'''\hat \pi_i = {e^{-0.0306152 - 0.016050LWT + 0.908519SMOKE + 0.489231PTL + 1.856358HT + 0.747161UI + 1.314579RACE2 + 0.860976RACE3}\over 1 + e^{-0.0306152 - 0.016050LWT + 0.908519SMOKE + 0.489231PTL + 1.856358HT + 0.747161UI + 1.314579RACE2 + 0.860976RACE3}} ''')


st.subheader('Gráfico da Probabilidade Estimada:')
st.pyplot(plot_probability(data))

    
    
st.subheader('Matriz de Confusão:')
st.pyplot(confusion_matrix_plot(modelo))


st.subheader('Logito Encontrado:')
st.latex(r'''\hat g(x_i) = -0.0306152 - 0.016050LWT + 0.908519SMOKE + 0.489231PTL + 1.856358HT + 0.747161UI + 1.314579RACE2 + 0.860976RACE3''')
#Intervalo de Confiança
st.subheader('Gráfico do Logito Estimado:')
st.pyplot(plot_logit(modelo,data))
    
    
   
#Residuo de Pearson:
residual_pearson(modelo,data)  


#Curva Roc:
st.subheader('Curva ROC:')
st.pyplot(roc_curve(modelo,data))

#Texto:
st.text('Este aplicativo foi um trabalho final da disciplina de Análise de Dados: Regressão Logística.\nOs dados foram obtidos no livro de HOSMER,D.; LEMESHOW, S. Applied logistic regression. 2nd ed. New York: Wiley, 2000. 375 p.')

#Streamlit
st.sidebar.header("Predição")
Name = st.sidebar.text_input("Digite o seu nome: ")
LWT = st.sidebar.slider("Selecione o seu peso do ultimo periodo menstrual em libras",1,200,1)
SMOKE = st.sidebar.selectbox("Você fuma ?",options = ['Sim','Não'])
PTL = st.sidebar.selectbox("Você tem histórico de trabalho de parto prematuro ?", options = ['Sim','Não'])
HT = st.sidebar.selectbox('Você tem hipertensão ?', options = ["Sim",'Não'])
UI = st.sidebar.selectbox("Você tem irritabilidade uterina ?",options = ['Sim','Não'])
RACE = st.sidebar.selectbox("Selecione a sua raça:",options = ["Branca",'Negra','Outra'])

b_race,o_race = 0,0
    
if PTL == 'Sim':
    PTL = 1
        
else:    
    PTL = 0
        
        
if HT == 'Sim':
    HT = 1
        
else:
    HT = 0
        
    
if b_race == 'Negra':
    b_race = 1
        
elif o_race == "Outra":
    o_race = 1
        
else:
    b_race,o_race = 0,0
        
if UI == "Sim":
        
    UI = 1
    
else:
    UI = 0   
    
if SMOKE == 'Sim':
    
    SMOKE = 1

else:
   
    SMOKE = 0      
  
if st.sidebar.button("Predizer"):
    
    input_data = np.asarray([LWT,SMOKE,PTL,HT,UI,b_race,o_race]).reshape((1,7))
    details = np.c_[np.ones((1,1)),input_data] 
    result = modelo.predict(sm.add_constant(details)) 
    st.sidebar.subheader('A {} tem {}% de chance de seu filho nascer com baixo peso'.format(Name, np.round(result*100 , 2)))        
 
#streamlit run proj.py






