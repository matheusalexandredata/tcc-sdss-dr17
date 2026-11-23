from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
import hashlib
import io
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
URL = 'https://www.kaggle.com/api/v1/datasets/download/fedesoriano/stellar-classification-dataset-sdss17?datasetVersionNumber=1'
CLASSES = ['GALAXY', 'QSO', 'STAR']
FEATURES = ['u_g', 'g_r', 'r_i', 'i_z', 'redshift']
HASH_CSV = 'faefcad9b950309282024dd351289ee14ae30f65a7b2ef32c9fa442da9dbe585'
SEED = 42


def load_data():
    arquivo = ROOT / 'dados' / 'sdss17.zip'
    if not arquivo.is_file():
        print('Baixando o SDSS17 do Kaggle...', flush=True)
        arquivo.parent.mkdir(exist_ok=True)
        temporario = arquivo.with_suffix('.tmp')
        try:
            with urlopen(Request(URL, headers={'User-Agent': 'Python'}), timeout=120) as resposta:
                temporario.write_bytes(resposta.read())
            leia_csv(temporario)
            temporario.replace(arquivo)
        except (OSError, BadZipFile, KeyError, ValueError) as erro:
            raise SystemExit('Baixe a versão 1 no link do README e salve o ZIP em dados/sdss17.zip.') from erro
        finally:
            temporario.unlink(missing_ok=True)
    df = pd.read_csv(io.BytesIO(leia_csv(arquivo)), dtype={'obj_ID': 'string', 'spec_obj_ID': 'string'})
    df['linha_csv'] = df.index + 2
    sem_sentinela = ~df[['u', 'g', 'r', 'i', 'z', 'redshift']].eq(-9999).any(axis=1)
    # As linhas 62670 e 62671 têm o mesmo bestObjID no SkyServer. Foi mantida a primeira.
    df = df.loc[sem_sentinela & df['linha_csv'].ne(62671)].reset_index(drop=True)
    assert df['class'].value_counts().to_dict() == {'GALAXY': 59444, 'STAR': 21593, 'QSO': 18961}
    for cor, a, b in [('u_g', 'u', 'g'), ('g_r', 'g', 'r'), ('r_i', 'r', 'i'), ('i_z', 'i', 'z')]:
        df[cor] = df[a] - df[b]
    X = df[FEATURES].to_numpy(dtype=float)
    y = df['class'].map({classe: i for i, classe in enumerate(CLASSES)}).to_numpy(dtype=int)
    treino, teste = train_test_split(np.arange(len(df)), test_size=.3, stratify=y, random_state=SEED)
    ajuste, validacao = train_test_split(treino, test_size=.2, stratify=y[treino], random_state=SEED)
    return df, X, y, treino, teste, ajuste, validacao


def leia_csv(arquivo):
    with ZipFile(arquivo) as zip_dados:
        csv = zip_dados.read('star_classification.csv')
    if hashlib.sha256(csv).hexdigest() != HASH_CSV:
        raise ValueError('O CSV não corresponde à versão usada no trabalho.')
    return csv


class EntradaMLP(TransformerMixin, BaseEstimator):
    """Padroniza as entradas. Quando selecionado, aplica asinh ao redshift antes da padronização."""
    def __init__(self, modo='identity'):
        self.modo=modo

    def fit(self,X,y=None):
        self.scaler_=StandardScaler().fit(self._transform_fixed(X))
        self.n_features_in_=np.asarray(X).shape[1]
        return self

    def _transform_fixed(self,X):
        array=np.array(X,dtype=float,copy=True)
        if self.modo=='asinh_0.001':
            array[:,4]=np.arcsinh(array[:,4]/.001)
        elif self.modo!='identity':
            raise ValueError(self.modo)
        return array

    def transform(self,X):
        return self.scaler_.transform(self._transform_fixed(X))


def candidates():
    result=[]
    architectures=[(32,),(64,),(128,),(64,32),(128,64),(64,32,16)]
    for mode in ['identity','asinh_0.001']:
        for layers in architectures:
            result.append(dict(layers=list(layers),activation='relu',alpha=.0001,
                learning_rate_init=.001,batch_size=256,preprocessing=mode,balanceado=False))
    extras=[
        ((64,32),'tanh',.0001,.001,256,'identity',False),
        ((128,64),'tanh',.0001,.001,256,'asinh_0.001',False),
        ((64,32),'relu',.001,.003,512,'asinh_0.001',False),
        ((128,64),'relu',.00001,.003,256,'asinh_0.001',False),
        ((64,32),'relu',.0001,.001,256,'asinh_0.001',True),
        ((128,64,32),'relu',.001,.003,512,'asinh_0.001',True),
    ]
    for layers,act,alpha,lr,batch,mode,balanced in extras:
        result.append(dict(layers=list(layers),activation=act,alpha=alpha,learning_rate_init=lr,
                           batch_size=batch,preprocessing=mode,balanceado=balanced))
    for i,c in enumerate(result,1):
        c['id'] = f'MLP{i:02d}'
    return result


def make_mlp(config,seed=SEED):
    return MLPClassifier(hidden_layer_sizes=tuple(config['layers']),activation=config['activation'],
        solver='adam',alpha=config['alpha'],batch_size=config['batch_size'],
        learning_rate_init=config['learning_rate_init'],early_stopping=False,shuffle=True,
        random_state=np.random.RandomState(seed),max_iter=1,tol=0.,n_iter_no_change=100000)


def metrics(y,pred,proba=None):
    report=classification_report(y,pred,labels=[0,1,2],target_names=CLASSES,output_dict=True,zero_division=0)
    result={'f1_macro':float(f1_score(y,pred,average='macro',zero_division=0)),
        'recall_macro':float(recall_score(y,pred,average='macro',zero_division=0)),
        'precision_macro':float(precision_score(y,pred,average='macro',zero_division=0)),
        'acuracia':float(accuracy_score(y,pred)),
        'acuracia_balanceada':float(balanced_accuracy_score(y,pred))}
    for c in CLASSES:
        result[f'recall_{c}']=float(report[c]['recall'])
        result[f'f1_{c}']=float(report[c]['f1-score'])
    if proba is not None:result['log_loss']=float(log_loss(y,proba,labels=[0,1,2]))
    return result


def rank_key(record):
    m=record['best_metrics']
    return (-m['f1_macro'],-m['recall_macro'],-m['precision_macro'],sum(record['config']['layers']),record['config']['id'])


def train_candidate(config,Xt,yt,Xv,yv,max_epochs,state=None,stop=False):
    if state is None:
        prep=EntradaMLP(config['preprocessing']).fit(Xt)
        state={'config':config,'preprocessor':prep,'current_model':make_mlp(config),
               'epoch':0,'best_epoch':0,'best_f1':-1.,'last_improvement':0,
               'history':[]}
    tx = state['preprocessor'].transform(Xt)
    vx = state['preprocessor'].transform(Xv)
    weights=compute_sample_weight('balanced',yt) if config['balanceado'] else None
    for epoch in range(state['epoch']+1,max_epochs+1):
        state['current_model'].partial_fit(tx,yt,classes=np.array([0,1,2]),sample_weight=weights)
        state['epoch']=epoch
        if epoch%5==0 or epoch==max_epochs:
            proba = state['current_model'].predict_proba(vx)
            pred = proba.argmax(axis=1)
            m=metrics(yv,pred,proba)
            state['history'].append({'epoch':epoch,'loss_treino':float(state['current_model'].loss_),**m})
            if m['f1_macro']>state['best_f1']:
                state['best_f1'] = m['f1_macro']
                state['best_epoch'] = epoch
                state['best_metrics']=m
            # Critério de paciência separado da preservação de qualquer melhor valor.
            if m['f1_macro']>state.get('patience_score',-1.)+.0001:
                state['patience_score']=m['f1_macro'];state['last_improvement']=epoch
            print(f"{config['id']} epoca {epoch:03d} valid F1={m['f1_macro']:.6f}",flush=True)
            if stop and epoch-state['last_improvement']>=30:
                break
    return state


def traditional_candidates():
    result=[]
    for C in [.1,1.,10.]:
        result.append(('LR',{'C':C},Pipeline([('scaler',StandardScaler()),
                       ('model',LogisticRegression(C=C,solver='lbfgs',max_iter=3000))])))
    for depth in [2,3,4,5,8,12,None]:
        result.append(('DT',{'max_depth':depth},DecisionTreeClassifier(max_depth=depth,random_state=SEED)))
    for count in [100,200]:
        for depth in [None,12,20]:
            result.append(('RF',{'n_estimators':count,'max_depth':depth},
                RandomForestClassifier(n_estimators=count,max_depth=depth,random_state=SEED,n_jobs=2)))
    return result


def selecionar(X, y, ajuste, validacao, so_cores):
    Xt, yt = X[ajuste], y[ajuste]
    Xv, yv = X[validacao], y[validacao]
    configs = candidates()
    if so_cores:
        # Sem redshift, pares que diferiam apenas pelo asinh passam a ser iguais.
        unicas = {}
        for config in configs:
            config['preprocessing'] = 'identity'
            chave = tuple(str(v) for k, v in config.items() if k != 'id')
            unicas.setdefault(chave, config)
        configs = list(unicas.values())

    estados = [train_candidate(c, Xt, yt, Xv, yv, 30) for c in configs]
    promovidas = sorted(estados, key=rank_key)[:3]
    for estado in promovidas:
        train_candidate(estado['config'], Xt, yt, Xv, yv, 200, state=estado, stop=True)
    mlp = min(estados, key=rank_key)

    tradicionais = {}
    for nome, params, modelo in traditional_candidates():
        modelo.fit(Xt, yt)
        medida = metrics(yv, modelo.predict(Xv))
        criterio = tuple(medida[k] for k in ('f1_macro', 'recall_macro', 'precision_macro'))
        if nome not in tradicionais or criterio > tradicionais[nome][0]:
            tradicionais[nome] = (criterio, params, clone(modelo))
    return tradicionais, mlp, estados


def ajustar_mlp(config, epocas, X, y):
    entrada = EntradaMLP(config['preprocessing']).fit(X)
    Xt = entrada.transform(X)
    modelo = make_mlp(config)
    pesos = compute_sample_weight('balanced', y) if config['balanceado'] else None
    for _ in range(epocas):
        modelo.partial_fit(Xt, y, classes=np.array([0, 1, 2]), sample_weight=pesos)
    return Pipeline([('entrada', entrada), ('model', modelo)])


def executar():
    df, X, y, treino, teste, ajuste, validacao = load_data()
    cores = X[:, :4].copy()
    predicoes = df.iloc[teste][['linha_csv', 'class']].copy()
    medidas, classes, matrizes, curvas, escolhas = {}, {}, {}, [], {}

    def avaliar(cenario, nome, modelo, entradas):
        chave = f'{cenario}_{nome}'
        probas = modelo.predict_proba(entradas[teste])
        pred = modelo.predict(entradas[teste])
        medidas[chave] = metrics(y[teste], pred, probas)
        classes[chave] = classification_report(y[teste], pred, labels=[0, 1, 2],
            target_names=CLASSES, zero_division=0, output_dict=True)
        matrizes[chave] = confusion_matrix(y[teste], pred, labels=[0, 1, 2]).tolist()
        predicoes[chave] = np.asarray(CLASSES)[pred]

    print('\nSeleção com redshift', flush=True)
    com, rede_com, estados_com = selecionar(X, y, ajuste, validacao, so_cores=False)
    print('\nSeleção com quatro cores', flush=True)
    sem, rede_sem, estados_sem = selecionar(cores, y, ajuste, validacao, so_cores=True)

    for cenario, entradas, tradicionais, rede in [
        ('com_redshift', X, com, rede_com),
        ('sem_redshift_configuracao_mantida', cores, com, rede_com),
        ('sem_redshift_nova_selecao', cores, sem, rede_sem),
    ]:
        config = rede['config'].copy()
        if cenario != 'com_redshift':
            config['preprocessing'] = 'identity'
        epocas = rede['best_epoch']
        escolhas[cenario] = {nome: params for nome, (_, params, _) in tradicionais.items()}
        escolhas[cenario]['MLP'] = {**config, 'epocas': epocas}
        for nome, (_, _, estimador) in tradicionais.items():
            ajustado = clone(estimador).fit(entradas[treino], y[treino])
            avaliar(cenario, nome, ajustado, entradas)
            if cenario == 'com_redshift' and nome == 'RF':
                importancia = dict(zip(FEATURES, ajustado.feature_importances_.tolist()))
        avaliar(cenario, 'MLP', ajustar_mlp(config, epocas, entradas[treino], y[treino]), entradas)

    config_controle = {**rede_com['config'], 'preprocessing': 'identity'}
    controle = ajustar_mlp(config_controle, rede_com['best_epoch'], X[treino], y[treino])
    avaliar('com_redshift_sem_asinh', 'MLP', controle, X)

    for cenario, estados in [('com_redshift', estados_com), ('sem_redshift', estados_sem)]:
        for estado in estados:
            curvas.extend({'cenario': cenario, 'modelo': estado['config']['id'], **h} for h in estado['history'])
    particoes = df[['linha_csv', 'class']].copy()
    particoes['conjunto'] = 'teste'
    particoes.loc[ajuste, 'conjunto'] = 'ajuste'
    particoes.loc[validacao, 'conjunto'] = 'validacao'

    saida = ROOT / 'resultados'
    saida.mkdir(exist_ok=True)
    particoes.to_csv(saida / 'particoes.csv', index=False)
    predicoes.to_csv(saida / 'predicoes.csv', index=False)
    pd.DataFrame(curvas).to_csv(saida / 'validacao_mlp.csv', index=False)
    tabela = pd.DataFrame.from_dict(medidas, orient='index')
    tabela.to_csv(saida / 'metricas.csv', index_label='modelo')
    resumo = {'configuracoes': escolhas, 'metricas_por_classe': classes,
              'matrizes_de_confusao': matrizes, 'importancia_random_forest': importancia}
    (saida / 'resultados.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    print((tabela[['f1_macro', 'acuracia', 'recall_QSO']] * 100).round(2).to_string())


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        executar()
