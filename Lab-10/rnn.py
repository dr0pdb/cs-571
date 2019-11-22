
# coding: utf-8

# In[2]:


import re
f = open("rnn_lab/Brown_train.txt", "r")

text_list=[]
annote_list=[]
text_word_set=set()
text_label_set=set()
for x in f:    
    mystr = x
#     print(x)
    wordList = x.split()
#     print(wordList)
    text_word=[]
    annote_word=[]
    for z in wordList:
        
        u=z.split('/')
        if not(u[1].isupper()):
            continue
            
        text_word.append(u[0])
        annote_word.append(u[1])
    

#     print(annote)
    text_list.append(text_word)
    annote_list.append(annote_word)
    for word in text_word:
        text_word_set.add(word)
    for word in annote_word:    
        text_label_set.add(word)


# In[3]:


len(text_word_set)


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


X_train, X_test, y_train, y_test=train_test_split(text_list,annote_list, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# In[6]:


X_train


# In[7]:


dict_label={}
dict_word={}


# In[8]:


num=1
for i in text_word_set:
    dict_word[i]=num
    num=num+1


# In[9]:


len(dict_word)


# In[10]:


num=0
for i in text_label_set:
    dict_label[i]=num
    num=num+1


# In[11]:


dict_label


# In[12]:


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
        text_to_num=[]
        class_to_num=[]
        l=0
        for j in self.text_data[index]:
            text_to_num.append((dict_word[j]))
            l=l+1
        for j in self.text_label[index]:    
            class_to_num.append(dict_label[j])

        return np.array(text_to_num),np.array(class_to_num)
train_loader=SequenceFolder( X_train,y_train)
validation_loader=SequenceFolder( X_val,y_val)
test_loader=SequenceFolder(X_test,y_test)


# In[10]:


for a,b in train_loader:
    print(a.shape,b.shape)


# In[13]:


class MLP(nn.Module):
    def __init__(self,vocab_size,embedding_dim,device):
        super(MLP, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=100,bidirectional=True).to(device)
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.fc1_1=nn.Linear(200, 150)
        self.fc1_2=nn.Linear(200,150)
#         self.bn1=nn.BatchNorm1d(num_features=600)
        
        self.relu=nn.ReLU(inplace=True)


        self.fc_emo=nn.Linear(200, 11)
        self.sigm=nn.Sigmoid()
        self.dropout1=nn.Dropout(0.2)
        self.dropout2=nn.Dropout(0.3)
        self.device=device

    def forward(self, x):
        h0=c0=torch.zeros(2, 1, 100).to(self.device)
        emb=self.embedding(x)
        emb=emb.permute(1,0,2)
        
        after_lstm= self.lstm(emb, (h0, c0))[0]
#         print(after_lstm.shape)
        return(self.fc_emo(after_lstm))


# In[15]:


import torch
# u=MLP(33588,100,'cuda:5').to('cuda:5')
# u(torch.zeros(1,20).long().to('cuda:5')).shape


# In[24]:



device='cuda:0'
model=MLP(33589,100,device).to(device)
# model3=lstm2.to(device)
criterion = nn.CrossEntropyLoss()
optim_params = [
#     {'params': model1.parameters(), 'lr':0.0001},
#     {'params': model2.parameters(), 'lr':0.001},
    {'params': model.parameters(), 'lr':0.0001}
        
]
optimizer = torch.optim.Adam(optim_params)


# In[28]:


for i in range(500):
        total_correct=0
        total=0
        confusion_matrix = torch.zeros(11, 11)
        u=0
        for GN,lab in (validation_loader):
#             gen_nam=GN[0]
            print(u)
            
#             seq=seq.permute(1,0,2)
#             print(seq.shape)
            seq=torch.tensor(GN)
            
            lab=torch.tensor(lab).to(device)
            inp=seq.to(device)
              
            
#             print(inp.shape,lab.shape)
            
#             h_x2=c_x2=torch.zeros(1, 4, 200).to(device)
#             output, (hn, cn) = model1(inp.float(), (h0, c0))  
#             print(output.shape)
#             output,(_,_)=model3(output,(h_x2,c_x2))
#             print(output.shape)
#             q1,q2,q3=output.shape
#             out_last=((output[q1-1]))
            
#             After_fc=model2(out_last)
            
            After_fc=model(inp.long().reshape(1,-1))
            After_fc=After_fc.squeeze(1)

            _, preds = torch.max(After_fc, 1)
            total_correct=total_correct+torch.sum(preds.long()==lab.long().to(device))
            total=total+preds.shape[0]
            
            for t, p in zip(lab.view(-1), preds.view(-1)):
 
                confusion_matrix[t.long(), p.long()] += 1
            u=u+1
        print(total_correct,total)
        print(confusion_matrix)
        
    
    
    
    
    
    
    
        loss_epoch=0
        loss_batch=0
        u=0
        for GN,lab in (train_loader):
            
#             gen_nam=GN[0]
            seq=torch.tensor(GN)
#             seq=seq.permute(1,0,2)
            lab=torch.tensor(lab).to(device)
            inp=seq.to(device)
#             print(inp.shape)

             

            
#             output, (hn, cn) = model1(inp.float(), (h0, c0)) 
#             h_x2=c_x2=torch.zeros(1, 4, 200).to(device)
#             output,(_,_)=model3(output,(h_x2,c_x2))
            

#             q1,q2,q3=output.shape
#             out_last=((output[q1-1]))
#             After_fc=model2(out_last)
            optimizer.zero_grad()
            After_fc=model(inp.long().reshape(1,-1))
            After_fc=After_fc.squeeze(1)
            loss = criterion(After_fc, lab).to(device)/64
            loss_batch=loss
            
            

            
            u=u+1
#             if u%4==0:
            loss_epoch=loss_epoch+loss_batch
            
            loss_batch.backward()
#             plot_grad_flow(model_CNN.named_parameters())
            optimizer.step()
            optimizer.zero_grad()
#             break
            
#         break        
        print(loss_epoch)


# In[29]:


confusion_matrix = torch.zeros(11, 11)
u=0
for GN,lab in (test_loader):
#             gen_nam=GN[0]
    print(u)

#             seq=seq.permute(1,0,2)
#             print(seq.shape)
    seq=torch.tensor(GN)

    lab=torch.tensor(lab).to(device)
    inp=seq.to(device)


#             print(inp.shape,lab.shape)

#             h_x2=c_x2=torch.zeros(1, 4, 200).to(device)
#             output, (hn, cn) = model1(inp.float(), (h0, c0))  
#             print(output.shape)
#             output,(_,_)=model3(output,(h_x2,c_x2))
#             print(output.shape)
#             q1,q2,q3=output.shape
#             out_last=((output[q1-1]))

#             After_fc=model2(out_last)

    After_fc=model(inp.long().reshape(1,-1))
    After_fc=After_fc.squeeze(1)

    _, preds = torch.max(After_fc, 1)
    total_correct=total_correct+torch.sum(preds.long()==lab.long().to(device))
    total=total+preds.shape[0]

    for t, p in zip(lab.view(-1), preds.view(-1)):

        confusion_matrix[t.long(), p.long()] += 1
    u=u+1
print(total_correct,total)
print(confusion_matrix)
        


# In[41]:


precision=[]
recall=[]
fscore=[]
for i in range(11):
    tp=confusion_matrix[i][i]
    pr=0
    rec=0
    for row in range(11):
        pr=pr+confusion_matrix[row][i]
    for col in range(11):
        rec=rec+confusion_matrix[i][col]
#     fsc=2/(1/pr+1/rec)
    precision.append(tp/pr)
    recall.append(tp/rec)
    fscore.append(2/(1/precision[i]+1/recall[i]))
        
        


# In[45]:


dict_label


# In[47]:


print(precision)


# In[48]:


recall


# In[49]:


fscore

