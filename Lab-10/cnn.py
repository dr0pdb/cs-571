
# coding: utf-8

# In[1]:


import re


# In[2]:


def split_wordtags(corpus, delimiter='/', start_word='*', stop_word='STOP', ngram_used=3):
    """
    Splits a corpus into a words vector and a tag vector
    :param corpus:
    :param delimiter:
    :param start_word:
    :param stop_word:
    :param ngram_used: Default=3 . # of ngrams to use. Will insert start and stop accordingly.
    :return:
    """
    tag_sentences = []
    word_sentences = []

    # for each sentence
    for sentence in corpus:
        # split on space
        word_list = n_gramer.explode(sentence)
        words = []
        tags = []
        for el in word_list:
            word, tag = el.rsplit(delimiter, 1)
            words.append(word)
            tags.append(tag)

        # Insert start and end token in each vector
        n_gramer.insert_start_end_tokens(words, start_word, stop_word, ngram_used)
        n_gramer.insert_start_end_tokens(tags, start_word, stop_word, ngram_used)

        words_sentence = ' '.join(words)
        tags_sentence = ' '.join(tags)

        tag_sentences.append(tags_sentence)
        word_sentences.append(words_sentence)

    return word_sentences, tag_sentences


# In[3]:


f = open("cnn_lab/waste_train.txt", "r")
train_class=[]
train_text=[]
train_word_set=set()
train_class_set=set()
for x in f:
    first, *middle, last = x.split()
    print(last)
    result = re.search('BOS (.*) EOS', x)
    output = re.sub(r'\d+', '', result.group(1))
    if not(str(last).find('#')==-1):
        continue
    mystr = output
    wordList = re.sub("[^\w]", " ",  mystr).split()
    train_text.append(wordList)
    train_class.append(last)
    train_class_set.add(last)
    for word in wordList:
        train_word_set.add(word)
    
  
    


# In[4]:


(train_text[0])


# In[5]:


f = open("cnn_lab/waste_test.txt", "r")
test_class=[]
test_text=[]
test_word_set=set()
test_class_set=set()
for x in f:
    first, *middle, last = x.split()
    print(last)
    result = re.search('BOS (.*) EOS', x)
    output = re.sub(r'\d+', '', result.group(1))
    if not(str(last).find('#')==-1):
        continue
    mystr = output
    wordList = re.sub("[^\w]", " ",  mystr).split()
    test_text.append(wordList)
    test_class.append(last)
    test_class_set.add(last)
    for word in wordList:
        test_word_set.add(word)


# In[6]:


text_word_set=set()
text_word_set=train_word_set.union(test_word_set)

len(text_word_set)


# In[7]:


text_class_set=test_class_set.union(train_class_set)


# In[8]:


text_class_set


# In[9]:


dict_label={}
dict_word={}


# In[10]:


num=1
for i in text_word_set:
    dict_word[i]=num
    num=num+1
dict_word


# In[11]:


dict_word['comes']


# In[12]:


num=0
for i in text_class_set:
    dict_label[i]=num
    num=num+1


# In[13]:


len(dict_label)


# In[14]:


import torch.utils.data as data
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
class SequenceFolder(data.Dataset):
    def __init__(self,train_text,label):
        self.text_data=train_text
        self.text_label=label
#         self.label_emo=label_emotion
    def __len__(self):
        return len(self.text_label)       

    def __getitem__(self, index):
        text_to_num=np.zeros((50))
        l=0
        for j in self.text_data[index]:
            text_to_num[l]=(dict_word[j])
            l=l+1
        class_to_num=dict_label[self.text_label[index]]

        return np.array(text_to_num),np.array(class_to_num)
t_d=SequenceFolder( train_text,train_class)
train_dataloader = DataLoader(t_d, batch_size=32,
                        shuffle=True, num_workers=4,drop_last=True)


# In[15]:


t_d=SequenceFolder(test_text,test_class)
test_dataloader = DataLoader(t_d, batch_size=32,
                        shuffle=True, num_workers=4,drop_last=True)


# In[16]:


for i,j in train_dataloader:
    print(i.shape)


# In[17]:


import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], embedding_dim))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], embedding_dim))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], embedding_dim))
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
#         print(cat.shape)
        return self.fc(cat)


# In[18]:


import torch
model=CNN(800,100,8,(3,3,3),18,0.2,None)
model(torch.zeros(32,30).long()).shape


# In[19]:


import torch
device='cuda:0'

# model1=Encoder(embedding_matrix,2,300,4,300,300,True,4).to(device) .to(device)
model=CNN(756,100,8,(3,3,3),18,0.2,None).to(device)
# model3=lstm2.to(device)
criterion = nn.CrossEntropyLoss()
optim_params = [
    {'params': model.parameters(), 'lr':0.001},
#     {'params': model2.parameters(), 'lr':0.0001},
#     {'params': model3.parameters(), 'lr':0.0001}
        
]
optimizer = torch.optim.Adam(optim_params)


# In[20]:


mod_list=[]
for i in range(500):

        total_correct=0
        total=0
        confusion_matrix = torch.zeros(18, 18)

        
        for GN,lab in (test_dataloader):
#             print(lab)
#             break
#             gen_nam=GN[0]
           
            
#             seq=seq.permute(1,0,2)
#             print(seq.shape)

#             print(seq.shape)    
            seq=GN
            
            lab=lab.to(device)
            inp=seq.to(device)
             
#             break

            After_fc=model(inp.long())
            
            
            _, preds = torch.max(After_fc, 1)
        
            total_correct=total_correct+torch.sum(preds.long()==lab.long().to(device))
            total=total+After_fc.shape[0]
            
            for t, p in zip(lab.view(-1), preds.view(-1)):
 
                confusion_matrix[t.long(), p.long()] += 1
                        
        print(total_correct.cpu().data,total)
        print(confusion_matrix)
#         print(confusion_matrix_DAC)
#         print(confusion_matrix_COMBO)
    
    
    
    
    
    
    
        loss_epoch=0
        loss_batch=0
        u=0

        for GN,lab in (train_dataloader):
            
#             gen_nam=GN[0]
            seq=GN
#             print(GN)
#             seq=seq.permute(1,0,2)
            inp=seq.to(device)
#             print(inp.shape)
            
         
            if inp.shape[1]==2:
                continue
            
#             output, (hn, cn) = model1(inp.float(), (h0, c0)) 
#             h_x2=c_x2=torch.zeros(1, 4, 200).to(device)
#             output,(_,_)=model3(output,(h_x2,c_x2))
            

#             q1,q2,q3=output.shape
#             out_last=((output[q1-1]))
#             After_fc=model2(out_last)
            lab=lab.to(device)
            optimizer.zero_grad()

            After_fc=model(inp.long())
#             After_fc_DAC,y=model2(inp.long())
#             After_fc_COMBO=model3(x,y)

            loss1 = criterion(After_fc, lab).to(device)#

            loss_batch=loss1

            
            u=u+1
#             if u%4==0:
            loss_epoch=loss_epoch+loss_batch
            
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
#             count=count+1
#             break
            
#         break
        print(loss_epoch.cpu().data)


# In[23]:


get_ipython().system('nvidia-smi')


# In[22]:


confusion_matrix.shape


# In[29]:


precision=[]
recall=[]
fscore=[]
for i in range(18):
    tp=confusion_matrix[i][i]
    pr=0
    rec=0
    for row in range(18):
        pr=pr+confusion_matrix[row][i]
    for col in range(18):
        rec=rec+confusion_matrix[i][col]
#     fsc=2/(1/pr+1/rec)
    precision.append(tp/pr)
    recall.append(tp/rec)
    fscore.append(2/(1/precision[i]+1/recall[i]))
        
        


# In[30]:


precision


# In[32]:


recall


# In[33]:


fscore


# In[31]:


dict_label

