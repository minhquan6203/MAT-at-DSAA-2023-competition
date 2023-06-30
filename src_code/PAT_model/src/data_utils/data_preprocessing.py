import pandas as pd

df_text=pd.read_csv('./nodes.tsv',delimiter='\t')

# Read train data
df = pd.read_csv('train.csv')
df['id1_text']=df['id1']
df['id2_text']=df['id2']
df['Label']=df['label']
df = df.drop(['id','label'], axis=1)

# Read test data
dft = pd.read_csv('testcsv')
dft['id1_text']=dft['id1']
dft['id2_text']=dft['id2']
dft = dft.drop('id', axis=1)

stop_words=['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is',
            'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'you',
            'I', 'me', 'my', 'we', 'our', 'us', 'they', 'their', 'them', 'this', 'these', 'those',
            'here', 'there', 'where', 'when', 'why', 'how', 'which', 'what', 'who', 'whom',
            'could', 'should', 'would', 'can', 'shall', 'may', 'might', 'must', 'do', 'did', 'does',
            'doing', 'done', 'had', 'having', 'has', 'have', 'having', 'but', 'if', 'or', 'nor', 'not',
            'because', 'since', 'as', 'until', 'while', 'though', 'although', 'after', 'before',
            'unless', 'whether', 'either', 'neither', 'rather', 'yet', 'so', 'than', 'too', 'also',
            'very', 'much', 'more', 'less', 'few', 'many', 'some', 'any', 'each', 'every', 'all', 'both',
            'either', 'neither', 'none', 'other', 'another', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'several', 'that', 'these', 'this', 'those', 'through', 'throughout', 'up', 'down',
            'into', 'out', 'on', 'off', 'over', 'under', 'above', 'below', 'between', 'among', 'at',
            'by', 'during', 'since', 'for', 'from', 'to', 'with', 'within', 'without',]

common_dataset=['infobox','name','united','image','states','new','national','species',
    'football','list','0','1','2','3','district','country','caption','state','city','his',
    'county','one','language','first','province','american','music','south','born','website',
     'area','school','time','team','world','university','album','imagesize','party','location',
     'year','birthdate','coordinates','ship','league','known','birthplace','north','most',
     'population','rock','film','rugby','number','india','two','type','union','de','family',
     'released','been','cup','season','war','series','2016']

sign=[',', '.', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>', '"', "'", 
      '`', '...', '--', '-', '_', '*', '@', '#', '$', '%', '^', '&', '+', '=', '/', '\\', 
      '|', '~', '``', "''", '“', '”', '‘', '’', '«', '»', '„', '‟', '‹', '›', '〝', '〞', 
      '‒', '–', '—', '―',    '•', '·', '⋅', '°']

all_stop_words=stop_words+common_dataset
map_text={}
for i in range(len(df_text)):
  for t in sign:
    df_text['text'][i]=str(df_text['text'][i]).lower().replace(t,'')
    words =' '.join(w for w in df_text['text'][i].split() if w not in all_stop_words)
  map_text[df_text['id'][i]]=words

#map text train set
df['id1_text']=df['id1_text'].map(map_text)
df['id2_text']=df['id2_text'].map(map_text)

#map text test set
dft['id1_text']=dft['id1_text'].map(map_text)
dft['id2_text']=dft['id2_text'].map(map_text)

#save
df.to_csv('./train_new.csv')
dft.to_csv('./test_new.csv')