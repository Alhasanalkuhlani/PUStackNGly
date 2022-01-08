import numpy as np
import pandas as pd 


n_AAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_AAC.txt",sep='\t',header = 'infer',index_col="#")
#n_AAINDEX=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_AAINDEX.txt",sep='\t',header = 'infer',index_col="#")
n_BINARY=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_BINARY.txt",sep='\t',header = 'infer',index_col="#")
n_BLOSUM62=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_BLOSUM62.txt",sep='\t',header = 'infer',index_col="#")
n_CKSAAGP=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_CKSAAGP.txt",sep='\t',header = 'infer',index_col="#")
n_CKSAAP=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_CKSAAP.txt",sep='\t',header = 'infer',index_col="#")
n_CTDC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_CTDC.txt",sep='\t',header = 'infer',index_col="#")
n_CTDD=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_CTDD.txt",sep='\t',header = 'infer',index_col="#")
n_CTDT=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_CTDT.txt",sep='\t',header = 'infer',index_col="#")
n_CTriad=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_CTriad.txt",sep='\t',header = 'infer',index_col="#")
n_DDE=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_DDE.txt",sep='\t',header = 'infer',index_col="#")
n_DPC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_DPC.txt",sep='\t',header = 'infer',index_col="#")
n_EAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_EAAC.txt",sep='\t',header = 'infer',index_col="#")
n_EGAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_EGAAC.txt",sep='\t',header = 'infer',index_col="#")
n_GAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_GAAC.txt",sep='\t',header = 'infer',index_col="#")
n_GDPC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_GDPC.txt",sep='\t',header = 'infer',index_col="#")
n_Geary=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_Geary.txt",sep='\t',header = 'infer',index_col="#")
n_GTPC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_GTPC.txt",sep='\t',header = 'infer',index_col="#")
n_KSCTriad=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_KSCTriad.txt",sep='\t',header = 'infer',index_col="#")
n_Moran=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_Moran.txt",sep='\t',header = 'infer',index_col="#")
n_NMBroto=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_NMBroto.txt",sep='\t',header = 'infer',index_col="#")
#n_TPC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_TPC.txt",sep='\t',header = 'infer',index_col="#")
n_SSEC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_SSEC.txt",sep='\t',header = 'infer',index_col="#")
n_SSEB=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_SSEB.txt",sep='\t',header = 'infer',index_col="#")
n_PSSM=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PSSM.txt",sep='\t',header = 'infer',index_col="#")
n_PAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PAAC.txt",sep='\t',header = 'infer',index_col="#")
n_APAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_APAAC.txt",sep='\t',header = 'infer',index_col="#")
n_ASA=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_ASA.txt",sep='\t',header = 'infer',index_col="#")


# In[3]:


n_TYPE1=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type1.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE2=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type2.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE3A=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type3A.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE3B=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type3B.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE4=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type4.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE5=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type5.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE6A=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type6A.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE6B=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type6B.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE6C=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type6C.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE7=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type7.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE8=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type8.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE9=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type9.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE10=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type10.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE11=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type11.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE12=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type12.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE13=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type13.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE14=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type14.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE15=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type15.txt",sep='\t',header = 'infer',index_col="#")
n_TYPE16=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/NDS_H_PseKRAAC_type16.txt",sep='\t',header = 'infer',index_col="#")


# In[4]:


p_TYPE1=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type1.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE2=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type2.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE3A=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type3A.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE3B=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type3B.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE4=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type4.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE5=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type5.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE6A=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type6A.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE6B=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type6B.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE6C=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type6C.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE7=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type7.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE8=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type8.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE9=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type9.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE10=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type10.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE11=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type11.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE12=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type12.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE13=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type13.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE14=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type14.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE15=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type15.txt",sep='\t',header = 'infer',index_col="#")
p_TYPE16=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PseKRAAC_type16.txt",sep='\t',header = 'infer',index_col="#")


# In[5]:


n_TYPE2.iloc[0:4,0:4]


# In[6]:


p_AAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_AAC.txt",sep='\t',header = 'infer',index_col="#")
#p_AAINDEX=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_AAINDEX.txt",sep='\t',header = 'infer',index_col="#")
p_BINARY=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_BINARY.txt",sep='\t',header = 'infer',index_col="#")
p_BLOSUM62=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_BLOSUM62.txt",sep='\t',header = 'infer',index_col="#")
p_CKSAAGP=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_CKSAAGP.txt",sep='\t',header = 'infer',index_col="#")
p_CKSAAP=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_CKSAAP.txt",sep='\t',header = 'infer',index_col="#")
p_CTDC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_CTDC.txt",sep='\t',header = 'infer',index_col="#")
p_CTDD=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_CTDD.txt",sep='\t',header = 'infer',index_col="#")
p_CTDT=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_CTDT.txt",sep='\t',header = 'infer',index_col="#")
p_CTriad=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_CTriad.txt",sep='\t',header = 'infer',index_col="#")
p_DDE=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_DDE.txt",sep='\t',header = 'infer',index_col="#")
p_DPC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_DPC.txt",sep='\t',header = 'infer',index_col="#")
p_EAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_EAAC.txt",sep='\t',header = 'infer',index_col="#")
p_EGAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_EGAAC.txt",sep='\t',header = 'infer',index_col="#")
p_GAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_GAAC.txt",sep='\t',header = 'infer',index_col="#")
p_GDPC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_GDPC.txt",sep='\t',header = 'infer',index_col="#")
p_Geary=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_Geary.txt",sep='\t',header = 'infer',index_col="#")
p_GTPC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_GTPC.txt",sep='\t',header = 'infer',index_col="#")
p_KSCTriad=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_KSCTriad.txt",sep='\t',header = 'infer',index_col="#")
p_Moran=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_Moran.txt",sep='\t',header = 'infer',index_col="#")
p_NMBroto=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_NMBroto.txt",sep='\t',header = 'infer',index_col="#")
#p_TPC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_TPC.txt",sep='\t',header = 'infer',index_col="#")
p_SSEC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_SSEC.txt",sep='\t',header = 'infer',index_col="#")
p_SSEB=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_SSEB.txt",sep='\t',header = 'infer',index_col="#")
p_PSSM=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PSSM.txt",sep='\t',header = 'infer',index_col="#")
p_PAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_PAAC.txt",sep='\t',header = 'infer',index_col="#")
p_APAAC=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_APAAC.txt",sep='\t',header = 'infer',index_col="#")
p_ASA=pd.read_csv("F:/study/PhD/dataset20210318/iFeature/PDS_H_ASA.txt",sep='\t',header = 'infer',index_col="#")


# In[7]:


# Function to return the constant value columns of a given DataFrame
def remove_constant_value_features(df):
    return [e for e in df.columns if df[e].nunique() == 1]


# In[8]:


AAC=pd.concat([p_AAC,n_AAC]).add_suffix('_AAC')
AAC.drop(remove_constant_value_features(AAC),axis=1,inplace=True)


# In[9]:


AAC.shape


# In[10]:


BINARY=pd.concat([p_BINARY,n_BINARY]).add_suffix('_BIN')
BINARY.drop(remove_constant_value_features(BINARY),axis=1,inplace=True)


# In[11]:


BINARY.shape


# In[12]:


BLOSUM62=pd.concat([p_BLOSUM62,n_BLOSUM62]).add_suffix('_BLO')
BLOSUM62.drop(remove_constant_value_features(BLOSUM62),axis=1,inplace=True)


# In[13]:


BLOSUM62.shape


# In[14]:


CKSAAGP=pd.concat([p_CKSAAGP,n_CKSAAGP]).add_suffix('_CKG')
CKSAAGP.drop(remove_constant_value_features(CKSAAGP),axis=1,inplace=True)


# In[15]:


CKSAAGP.shape


# In[16]:


CKSAAP=pd.concat([p_CKSAAP,n_CKSAAP]).add_suffix('_CKS')
CKSAAP.drop(remove_constant_value_features(CKSAAP),axis=1,inplace=True)


# In[17]:


CKSAAP.shape


# In[18]:


CTDC=pd.concat([p_CTDC,n_CTDC]).add_suffix('_CTC')
CTDC.drop(remove_constant_value_features(CTDC),axis=1,inplace=True)


# In[19]:


CTDC.shape


# In[20]:


CTDD=pd.concat([p_CTDD,n_CTDD]).add_suffix('_CTD')
CTDD.drop(remove_constant_value_features(CTDD),axis=1,inplace=True)


# In[21]:


CTDD.shape


# In[22]:


CTDT=pd.concat([p_CTDT,n_CTDT]).add_suffix('_CTT')
CTDT.drop(remove_constant_value_features(CTDT),axis=1,inplace=True)


# In[23]:


CTDT.shape


# In[24]:


CTriad=pd.concat([p_CTriad,n_CTriad]).add_suffix('_CTR')
CTriad.drop(remove_constant_value_features(CTriad),axis=1,inplace=True)


# In[25]:


CTriad.shape


# In[26]:


DDE=pd.concat([p_DDE,n_DDE]).add_suffix('_DDE')
DDE.drop(remove_constant_value_features(DDE),axis=1,inplace=True)


# In[27]:


DDE.shape


# In[28]:


DPC=pd.concat([p_DPC,n_DPC]).add_suffix('_DPC')
DPC.drop(remove_constant_value_features(DPC),axis=1,inplace=True)


# In[29]:


DPC.shape


# In[30]:


EAAC=pd.concat([p_EAAC,n_EAAC]).add_suffix('_EAA')
EAAC.drop(remove_constant_value_features(EAAC),axis=1,inplace=True)


# In[31]:


EAAC.shape


# In[32]:


EGAAC=pd.concat([p_EGAAC,n_EGAAC]).add_suffix('_EGA')
EGAAC.drop(remove_constant_value_features(EGAAC),axis=1,inplace=True)


# In[33]:


EGAAC.shape


# In[34]:


GAAC=pd.concat([p_GAAC,n_GAAC]).add_suffix('_GAA')
GAAC.drop(remove_constant_value_features(GAAC),axis=1,inplace=True)


# In[35]:


GAAC.shape


# In[36]:


GDPC=pd.concat([p_GDPC,n_GDPC]).add_suffix('_GDP')
GDPC.drop(remove_constant_value_features(GDPC),axis=1,inplace=True)


# In[37]:


GDPC.shape


# In[38]:


Geary=pd.concat([p_Geary,n_Geary]).add_suffix('_GEA')
Geary.drop(remove_constant_value_features(Geary),axis=1,inplace=True)


# In[39]:


Geary.shape


# In[40]:


GTPC=pd.concat([p_GTPC,n_GTPC]).add_suffix('_GTP')
GTPC.drop(remove_constant_value_features(GTPC),axis=1,inplace=True)


# In[41]:


GTPC.shape


# In[42]:


KSCTriad=pd.concat([p_KSCTriad,n_KSCTriad]).add_suffix('_KSC')
KSCTriad.drop(remove_constant_value_features(KSCTriad),axis=1,inplace=True)


# In[43]:


KSCTriad.shape


# In[44]:


Moran=pd.concat([p_Moran,n_Moran]).add_suffix('_MOR')
Moran.drop(remove_constant_value_features(Moran),axis=1,inplace=True)


# In[45]:


Moran.shape


# In[46]:


NMBroto=pd.concat([p_NMBroto,n_NMBroto]).add_suffix('_NMB')
NMBroto.drop(remove_constant_value_features(NMBroto),axis=1,inplace=True)


# In[47]:


NMBroto.shape


# In[48]:


PSSM=pd.concat([p_PSSM,n_PSSM]).add_suffix('_PSS')
PSSM.drop(remove_constant_value_features(PSSM),axis=1,inplace=True)


# In[49]:


PSSM.shape


# In[50]:


SSEB=pd.concat([p_SSEB,n_SSEB]).add_suffix('_SSB')
SSEB.drop(remove_constant_value_features(SSEB),axis=1,inplace=True)


# In[51]:


SSEB.shape


# In[52]:


SSEC=pd.concat([p_SSEC,n_SSEC]).add_suffix('_SSC')
SSEC.drop(remove_constant_value_features(SSEC),axis=1,inplace=True)


# In[53]:


SSEC.shape


# In[54]:


TYPE1=pd.concat([p_TYPE1,n_TYPE1]).add_suffix('_TP1')
TYPE1.drop(remove_constant_value_features(TYPE1),axis=1,inplace=True)


# In[55]:


TYPE1.shape


# In[56]:


TYPE2=pd.concat([p_TYPE2,n_TYPE2]).add_suffix('_TP2')
TYPE2.drop(remove_constant_value_features(TYPE2),axis=1,inplace=True)


# In[57]:


TYPE2.shape


# In[58]:


TYPE3A=pd.concat([p_TYPE3A,n_TYPE3A]).add_suffix('_T3A')
TYPE3A.drop(remove_constant_value_features(TYPE3A),axis=1,inplace=True)


# In[59]:


TYPE3A.shape


# In[60]:


TYPE3B=pd.concat([p_TYPE3B,n_TYPE3B]).add_suffix('_T3B')
TYPE3B.drop(remove_constant_value_features(TYPE3B),axis=1,inplace=True)


# In[61]:


TYPE3B.shape


# In[62]:


TYPE4=pd.concat([p_TYPE4,n_TYPE4]).add_suffix('_TP4')
TYPE4.drop(remove_constant_value_features(TYPE4),axis=1,inplace=True)


# In[63]:


TYPE4.shape


# In[64]:


TYPE5=pd.concat([p_TYPE5,n_TYPE5]).add_suffix('_TP5')
TYPE5.drop(remove_constant_value_features(TYPE5),axis=1,inplace=True)


# In[65]:


TYPE5.shape


# In[66]:


TYPE6A=pd.concat([p_TYPE6A,n_TYPE6A]).add_suffix('_T6A')
TYPE6A.drop(remove_constant_value_features(TYPE6A),axis=1,inplace=True)


# In[67]:


TYPE6A.shape


# In[68]:


TYPE6B=pd.concat([p_TYPE6B,n_TYPE6B]).add_suffix('_T6B')
TYPE6B.drop(remove_constant_value_features(TYPE6B),axis=1,inplace=True)


# In[69]:


TYPE6B.shape


# In[70]:


TYPE6C=pd.concat([p_TYPE6C,n_TYPE6C]).add_suffix('_T6C')
TYPE6C.drop(remove_constant_value_features(TYPE6C),axis=1,inplace=True)


# In[71]:


TYPE6C.shape


# In[72]:


TYPE7=pd.concat([p_TYPE7,n_TYPE7]).add_suffix('_TP7')
TYPE7.drop(remove_constant_value_features(TYPE7),axis=1,inplace=True)


# In[73]:


TYPE7.shape


# In[74]:


TYPE8=pd.concat([p_TYPE8,n_TYPE8]).add_suffix('_TP8')
TYPE8.drop(remove_constant_value_features(TYPE8),axis=1,inplace=True)


# In[75]:


TYPE8.shape


# In[76]:


TYPE9=pd.concat([p_TYPE9,n_TYPE9]).add_suffix('_TP9')
TYPE9.drop(remove_constant_value_features(TYPE9),axis=1,inplace=True)


# In[77]:


TYPE9.shape


# In[78]:


TYPE10=pd.concat([p_TYPE10,n_TYPE10]).add_suffix('_T10')
TYPE10.drop(remove_constant_value_features(TYPE10),axis=1,inplace=True)


# In[79]:


TYPE10.shape


# In[80]:


TYPE11=pd.concat([p_TYPE11,n_TYPE11]).add_suffix('_T11')
TYPE11.drop(remove_constant_value_features(TYPE11),axis=1,inplace=True)


# In[81]:


TYPE11.shape


# In[82]:


TYPE12=pd.concat([p_TYPE12,n_TYPE12]).add_suffix('_T12')
TYPE12.drop(remove_constant_value_features(TYPE12),axis=1,inplace=True)


# In[83]:


TYPE12.shape


# In[84]:


TYPE13=pd.concat([p_TYPE13,n_TYPE13]).add_suffix('_T13')
TYPE13.drop(remove_constant_value_features(TYPE13),axis=1,inplace=True)


# In[85]:


TYPE13.shape


# In[86]:


TYPE14=pd.concat([p_TYPE14,n_TYPE14]).add_suffix('_T14')
TYPE14.drop(remove_constant_value_features(TYPE14),axis=1,inplace=True)


# In[87]:


TYPE14.shape


# In[88]:


TYPE15=pd.concat([p_TYPE15,n_TYPE15]).add_suffix('_T15')
TYPE15.drop(remove_constant_value_features(TYPE15),axis=1,inplace=True)


# In[89]:


TYPE15.shape


# In[90]:


TYPE16=pd.concat([p_TYPE16,n_TYPE16]).add_suffix('_T16')
TYPE16.drop(remove_constant_value_features(TYPE16),axis=1,inplace=True)


# In[91]:


TYPE16.shape


# In[92]:


PAAC=pd.concat([p_PAAC,n_PAAC]).add_suffix('_PAA')
PAAC.drop(remove_constant_value_features(PAAC),axis=1,inplace=True)


# In[93]:


PAAC.shape


# In[94]:


APAAC=pd.concat([p_PAAC,n_PAAC]).add_suffix('_APA')
APAAC.drop(remove_constant_value_features(APAAC),axis=1,inplace=True)


# In[95]:


APAAC.shape


# In[96]:


ASA=pd.concat([p_ASA,n_ASA]).add_suffix('_ASA')
ASA.drop(remove_constant_value_features(ASA),axis=1,inplace=True)


# In[97]:


ASA.shape


# In[100]:


train_data=pd.concat([AAC,BINARY,BLOSUM62,CKSAAGP,CKSAAP,CTDC,CTDD,CTDT,
         CTriad,DDE,DPC,EAAC,EGAAC,GAAC,GDPC,Geary,GTPC,
         KSCTriad,Moran,NMBroto,PSSM,SSEB,SSEC,ASA,
                     TYPE1,TYPE2,TYPE3A,TYPE3B,TYPE4,TYPE5,TYPE6A,TYPE6B,TYPE6C,TYPE7,
                     TYPE8,TYPE9,TYPE10,TYPE11,TYPE12,TYPE13,TYPE14,TYPE15,TYPE16,PAAC,APAAC],axis=1)




train_data.shape#




#Standard Scale for the data
from sklearn.preprocessing import StandardScaler
scaled_st = StandardScaler().fit_transform(train_data)
train_data_scaled=pd.DataFrame(scaled_st, index=train_data.index, columns=train_data.columns)

x_pos=train_data_scaled.iloc[0:n_p,]
x_neg=train_data_scaled.iloc[n_p:train_data.shape[0],]



x_train=pd.concat([x_pos,x_neg])
y_train=np.repeat([1,0], [n_p,n_n])

X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.15, random_state=333,stratify=y_train)


