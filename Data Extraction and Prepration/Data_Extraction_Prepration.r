library(readtext)
library(stringr)
library(rvest)


ng=readtext('datasets/UniProt_Glycoproteins.fasta')
head(ng)
#spilt the string
ngs=unlist(strsplit(ng$text, ">"))
ngs=ngs[-1]
#get the protein names
pnames=gsub(".*[:|:](.+)[:|].*", "\\1", ngs )



dsg=data.frame(stringsAsFactors = F)
dsg1=data.frame(stringsAsFactors = F)
dsg_all=data.frame(stringsAsFactors = F)
for(i in (1:length(pnames)))
{
  
  #Specifying the url for desired website to be scraped
  url <- paste('https://www.uniprot.org/uniprot/',pnames[i],sep ='')
  
  #Reading the HTML code from the website
  webpage <- read_html(url)
  #html_nodes(webpage,'#aaMod_section :nth-child(1)')
  rank_data_html <- html_nodes(webpage,'table#aaMod_section.featureTable')
  #rank_data <- html_text(rank_data_html)
  
  #Let's have a look at the rankings
  #head(rank_data)
  #?html_table()
  d=html_table( rank_data_html,header = T, fill = TRUE)
  dp=d[[1]][,1:3]
  
  dsg=rbind(dsg,data.frame(Pname=pnames[i],location= dp[str_detect(dp$`Feature key`,'Glycosylationi') & 
                                                          str_detect(dp$DescriptionActions,'N-linked') & str_detect(dp$DescriptionActions,'Publication') ,2],stringsAsFactors = F))
  dsg1=rbind(dsg1,data.frame(Pname=pnames[i],location= dp[str_detect(dp$`Feature key`,'Glycosylationi') & 
                                                          str_detect(dp$DescriptionActions,'N-linked') & (str_detect(dp$DescriptionActions,'Publication')| str_detect(dp$DescriptionActions,'Sequence analysis')) ,2],stringsAsFactors = F))

  dsg_all= rbind(dsg_all,data.frame(Pname=pnames[i],location= dp[str_detect(dp$`Feature key`,'Glycosylationi') & 
                                                               str_detect(dp$DescriptionActions,'N-linked')  ,2],stringsAsFactors = F))
  
}



seqs=gsub("\n","",gsub(".*_HUMAN(.+)$", "\\1", ngs ))

wstream=12
ws=25
# generate all S seqences with thier locations
n=length(pnames)
alldataS=data.frame(stringsAsFactors=FALSE)
for(i in (1:n))
{
  
  loc=vector()
  subseq=vector()
  status=vector()
  for (s in seqs)
  {
    if (is.null(s) ) next
    j=1
    loc=vector()
    subseq=vector()
    loc=do.call(rbind,str_locate_all(seqs[i],'N'))[,1]
    for (l in loc)
    {
      subseq[j]=substr(seqs[i],l-wstream,l+wstream)
      j=j+1
    }
    
    
  }
  if (length(loc)==0 ) next
  alldataS=rbind(alldataS,data.frame(Pname=pnames[i],location=loc,Seqence=subseq, stringsAsFactors=FALSE))
}


unip=dsg
head(unip)
uniall=dsg_all
head(uniall)

mrg1=merge(alldataS,unip,by.x = c('Pname','location'),by.y = c('Pname','Position.s.'),all  = F)
mrg_all=merge(alldataS,uniall,by.x = c('Pname','location'),by.y = c('Pname','Position.s.'),all  = F)
dim(mrg1[nchar(as.character(mrg1$Seqence))<ws,])
'%notin%' <- function(x,y)!('%in%'(x,y))
negt=alldataS[ alldataS$Seqence%notin% mrg1$Seqence,]
negt=alldataS[ (alldataS$Seqence%notin% mrg_all$Seqence),]
negt=negt[nchar(as.character(negt$Seqence))==ws,]
sum(grepl('N[A-Z]T|N[A-Z]S',negt$Seqence))
post=mrg1[nchar(as.character(mrg1$Seqence))==ws,]
#preprocessing
dsgsub=post
uniq_Pseq=unique(dsgsub$Seqence)
pds=data.frame(stringsAsFactors = F)
for (s in uniq_Pseq)
{
  pds=rbind(pds,data.frame(Pname=do.call(paste, c(as.list(dsgsub[dsgsub$Seqence==s,]$Pname), sep = "_"))
                           ,location=do.call(paste, c(as.list(dsgsub[dsgsub$Seqence==s,]$location), sep = "_"))
                           ,Seqence=s,stringsAsFactors = F))
}





write.csv(pds,file="PositiveSamples25_unique")

dsgsub=negt
uniq_Pseq=unique(dsgsub$Seqence)
nds=data.frame(stringsAsFactors = F)
for (s in uniq_Pseq)
{
  nds=rbind(nds,data.frame(Pname=do.call(paste, c(as.list(dsgsub[dsgsub$Seqence==s,]$Pname), sep = "_"))
                           ,location=do.call(paste, c(as.list(dsgsub[dsgsub$Seqence==s,]$location), sep = "_"))
                           ,Seqence=s,stringsAsFactors = F))
}

write.csv(nds,file="NegativeSamples25_unique")


library(seqinr)
np=dim(pds)[1]
for(i in (1:np ))
{
  write.fasta(pds$Seqence[i], paste(pds$Pname[i],'_',pds$location[i],sep =''), "PositiveSamples25_unique.fasta", open = "a", nbchar = 60, as.string = F)
  
}


nn=dim(nds)[1]

library(seqinr)
for(i in (1:nn ))
{
  write.fasta(nds$Seqence[i], paste(nds$Pname[i],'_',nds$location[i],sep =''), "NegativeSamples25_unique.fasta", open = "a", nbchar = 60, as.string = F)
  
}




