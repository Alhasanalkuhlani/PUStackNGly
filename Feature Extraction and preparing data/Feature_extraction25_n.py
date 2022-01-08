#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
import shutil
import scipy
import argparse
import collections
import platform
import math
import re
import numpy 
import pandas


# In[9]:


python iFeature.py --help


# In[10]:


python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type AAC --out datasets/extracted features/NDS_H_AAC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type EAAC --out datasets/extracted features/NDS_H_EAAC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type CKSAAP --out datasets/extracted features/NDS_H_CKSAAP.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type DPC --out datasets/extracted features/NDS_H_DPC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type DDE --out datasets/extracted features/NDS_H_DDE.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type GAAC --out datasets/extracted features/NDS_H_GAAC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type EGAAC --out datasets/extracted features/NDS_H_EGAAC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type CKSAAGP --out datasets/extracted features/NDS_H_CKSAAGP.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type GDPC --out datasets/extracted features/NDS_H_GDPC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type GTPC --out datasets/extracted features/NDS_H_GTPC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type BINARY --out datasets/extracted features/NDS_H_BINARY.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type BLOSUM62 --out datasets/extracted features/NDS_H_BLOSUM62.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type CTDC --out datasets/extracted features/NDS_H_CTDC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type CTDT --out datasets/extracted features/NDS_H_CTDT.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type CTDD --out datasets/extracted features/NDS_H_CTDD.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type CTriad --out datasets/extracted features/NDS_H_CTriad.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type KSCTriad --out datasets/extracted features/NDS_H_KSCTriad.txt


# In[11]:


python codes/Moran.py --file datasets/out_UnlabeledSamples25_unique.fasta  --nlag 24 --out datasets/extracted features/NDS_H_Moran.txt
python codes/NMBroto.py --file datasets/out_UnlabeledSamples25_unique.fasta  --nlag 24 --out datasets/extracted features/NDS_H_NMBroto.txt
python codes/Geary.py --file datasets/out_UnlabeledSamples25_unique.fasta  --nlag 24 --out datasets/extracted features/NDS_H_Geary.txt


# In[12]:


#change the lambdaValue to 20 before run this
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type PAAC --out datasets/extracted features/NDS_H_PAAC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type APAAC --out datasets/extracted features/NDS_H_APAAC.txt


# In[13]:


python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type AAC --out datasets/extracted features/PDS_H_AAC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type EAAC --out datasets/extracted features/PDS_H_EAAC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type CKSAAP --out datasets/extracted features/PDS_H_CKSAAP.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type DPC --out datasets/extracted features/PDS_H_DPC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type DDE --out datasets/extracted features/PDS_H_DDE.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type GAAC --out datasets/extracted features/PDS_H_GAAC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type EGAAC --out datasets/extracted features/PDS_H_EGAAC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type CKSAAGP --out datasets/extracted features/PDS_H_CKSAAGP.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type GDPC --out datasets/extracted features/PDS_H_GDPC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type GTPC --out datasets/extracted features/PDS_H_GTPC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type BINARY --out datasets/extracted features/PDS_H_BINARY.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type BLOSUM62 --out datasets/extracted features/PDS_H_BLOSUM62.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type CTDC --out datasets/extracted features/PDS_H_CTDC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type CTDT --out datasets/extracted features/PDS_H_CTDT.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type CTDD --out datasets/extracted features/PDS_H_CTDD.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type CTriad --out datasets/extracted features/PDS_H_CTriad.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type KSCTriad --out datasets/extracted features/PDS_H_KSCTriad.txt

python codes/Moran.py --file datasets/out_PositiveSamples25_unique.fasta  --nlag 24 --out datasets/extracted features/PDS_H_Moran.txt
python codes/NMBroto.py --file datasets/out_PositiveSamples25_unique.fasta  --nlag 24 --out datasets/extracted features/PDS_H_NMBroto.txt
python codes/Geary.py --file datasets/out_PositiveSamples25_unique.fasta  --nlag 24 --out datasets/extracted features/PDS_H_Geary.txt


# In[14]:


python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type PAAC --out datasets/extracted features/PDS_H_PAAC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type APAAC --out datasets/extracted features/PDS_H_APAAC.txt


# In[16]:


python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type SSEC --path datasets/extracted features/pss2_25  --out datasets/extracted features/NDS_H_SSEC.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type SSEB --path datasets/extracted features/pss2_25  --out datasets/extracted features/NDS_H_SSEB.txt
python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type PSSM --path datasets/extracted features/pssm_25  --out datasets/extracted features/NDS_H_PSSM.txt

python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type SSEC --path datasets/extracted features/pss2_25  --out datasets/extracted features/PDS_H_SSEC.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type SSEB --path datasets/extracted features/pss2_25  --out datasets/extracted features/PDS_H_SSEB.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type PSSM --path datasets/extracted features/pssm_25  --out datasets/extracted features/PDS_H_PSSM.txt


# In[8]:


python iFeature.py --file datasets/out_UnlabeledSamples25_unique.fasta --type ASA --path ProteinFeatureExtractionTools/spineXpublic/spXout/  --out datasets/extracted features/NDS_H_ASA.txt
python iFeature.py --file datasets/out_PositiveSamples25_unique.fasta --type ASA --path ProteinFeatureExtractionTools/spineXpublic/spXout/  --out datasets/extracted features/PDS_H_ASA.txt


python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type1 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type1.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type2 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type2.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type3A --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type3A.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type3B --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type3B.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type4 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type4.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type5 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 4 --out /extracted features/NDS_H_PseKRAAC_type5.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type6A --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type6A.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type6B --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type6B.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type6C --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type6C.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type7 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type7.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type8 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type8.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type9 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type9.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type10 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type10.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type11 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type11.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type12 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type12.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type13 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 4 --out /extracted features/NDS_H_PseKRAAC_type13.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type14 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type14.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type15 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type15.txt
python iFeaturePseKRAAC.py --file /out_UnlabeledSamples25_unique.fasta --type type16 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/NDS_H_PseKRAAC_type16.txt

python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type1 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type1.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type2 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type2.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type3A --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type3A.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type3B --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type3B.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type4 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type4.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type5 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 4 --out /extracted features/PDS_H_PseKRAAC_type5.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type6A --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type6A.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type6B --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type6B.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type6C --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type6C.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type7 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type7.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type8 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type8.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type9 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type9.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type10 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type10.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type11 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type11.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type12 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type12.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type13 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 4 --out /extracted features/PDS_H_PseKRAAC_type13.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type14 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type14.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type15 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type15.txt
python iFeaturePseKRAAC.py --file /out_PositiveSamples25_unique.fasta --type type16 --subtype lambda-correlation --ktuple 2 --gap_lambda 2 --raactype 5 --out /extracted features/PDS_H_PseKRAAC_type16.txt




