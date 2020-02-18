
# coding: utf-8

# In[ ]:

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

AG=dict()
AG["path"]="ag_news"
AG["word_min"]=0
AG["num_classes"]=4
AG["max_length"]=256

DBP=dict()
DBP["path"]="dbpedia"
DBP["word_min"]=5
DBP["num_classes"]=14
DBP["max_length"]=256

Yah_A=dict()
Yah_A["path"]="yahoo_answers"
Yah_A["word_min"]=5
Yah_A["num_classes"]=10
Yah_A["max_length"]=512

Amz_F=dict()
Amz_F["path"]="amazon_review_full"
Amz_F["word_min"]=10
Amz_F["num_classes"]=5
Amz_F["max_length"]=256

Amz_P=dict()
Amz_P["path"]="amazon_review_polarity"
Amz_P["word_min"]=10
Amz_P["num_classes"]=2
Amz_P["max_length"]=256

DataName=dict()
DataName["AG"]=AG
DataName["DBP"]=DBP
DataName["Yah. A."]=Yah_A
DataName["Amz F."]=Amz_F
DataName["Amz P."]=Amz_P


def read_data(path):
    print("Loading Data")
    train_ = open("/content/drive/My Drive/"+path+"_csv_train.txt",'r',encoding='utf-8')
    train_raw = []
    for lines in train_.readlines():
        lines=lines.split('\t')
        train_raw.append([lines[0].lower(),int(lines[1])])


    test_ = open("/content/drive/My Drive/"+path+"_csv_test.txt",'r',encoding='utf-8')
    test_raw = []
    for lines in test_.readlines():
        lines=lines.split('\t')
        test_raw.append([lines[0].lower(),int(lines[1])])

    random.shuffle(train_raw)
    
    return train_raw, test_raw

def dictionary(train,min_=1):
    print("Word dictionary is being made")    
    
    word_dict=dict()
    freq_dict=dict()

    word_dict["#PAD"]=0
    word_dict["#UNK"]=1

    for lines in train:
        sentence = lines[0].split()

        for word in sentence:
            if not word in word_dict:
                word_dict[word]=len(word_dict)
                freq_dict[word]=1
            else:
                freq_dict[word]+=1
                
                
    word_min=dict()
    word_min["#PAD"] = 0
    word_min["#UNK"]=1
    for i in freq_dict:
        if freq_dict[i]<=min_:
            continue
        else:
            word_min[i]=len(word_min)
    
    return word_min


def train_valid(train,valid_rate=0.1):
    print("Split train - dev")

    random.shuffle(train) # For making data sequences randomly
    train_ = train[:int(len(train_raw)*(1-valid_rate))]
    valid_ = train[int(len(train_raw)*(1-valid_rate)):]

    return train_, valid_


def split_padding(data,length,word_dict):
    length=length-4

    train_index=[]
    for lines in data:
        sentence=lines[0].split() # sentence split
        label = lines[1]-1 # because of difference between label and index

        edit=[]
        for word in sentence:
            if word in word_dict:
                edit.append(word_dict[word])
            else:
                edit.append(word_dict["#UNK"])
             
        train_index.append([[0]*4+edit[:length]+[0]*(length-len(edit)),label])
    
    return train_index
  

class EXAM(nn.Module):
    def __init__(self,voca_size,embedding_size,region_size,num_classes,max_length):
        super(EXAM, self).__init__()

        # Parameter
        self.voca_size = voca_size
        self.embedding_size = embedding_size
        self.region_size = region_size
        self.num_classes = num_classes
        self.max_length = max_length
        
        self.region_radius = self.region_size//2
        self.max_length = max_length
        self.entire = max_length-2*self.region_radius
        
        # Word-level Encoder
        self.embedding_region = nn.Embedding(self.voca_size,self.region_size*self.embedding_size,padding_idx=0)
        self.embedding_region.weight.data.uniform_(-0.01,0.01)

        self.embedding_word = nn.Embedding(self.voca_size,self.embedding_size,padding_idx=0)
        self.embedding_word.weight.data.uniform_(-0.01,0.01)

        # Interaction Layer
        self.interaction = nn.Linear(self.embedding_size,self.num_classes,bias=False)
        self.interaction.weight.data.uniform_(-0.01,0.01)

        # Aggreation Layer
        self.aggregation1 = nn.Linear(self.entire,self.max_length*2)
        torch.nn.init.xavier_uniform_(self.aggregation1.weight)

        self.aggregation2 = nn.Linear(self.max_length*2,1)
        torch.nn.init.xavier_uniform_(self.aggregation2.weight)        


    def forward(self,train_input):        
        batch_size=train_input.size()[0] # batch size
        length=train_input.size()[1]     # max_length

        
        # target word
        trimed_seq=train_input[:,self.region_radius:length-self.region_radius]                # [batch, length-region_size+1]
        K = self.embedding_region(trimed_seq)                                       # [batch, length-region_size+1, Region_size*embedding_size]
        K = K.reshape(batch_size,-1,self.region_size,self.embedding_size)                     # [batch, length-region_size+1, Region_size, embedding_size]

        # Region word
        E = torch.cat([train_input[:,nu-self.region_radius:nu-self.region_radius+self.region_size].unsqueeze(0)                        for nu in range(self.region_radius,length-self.region_radius)],dim=0)       # [length-region_size+1, batch_size, region_size]
    
        E = E.transpose(1,0)                                                         # [batch_size, length-region_size+1, region_size]
        E = E.transpose(2,1)                                                         # [batch_size, region_size, length-region_size+1]
        E = self.embedding_word(E)
        E = E.reshape(batch_size,self.entire,self.region_size,self.embedding_size)                  # [batch, length-region_size+1, region_size, embedding_size]


        Inter=torch.max(E*K,dim=2)[0]
        Inter=F.relu(Inter)
        Inter=Inter.reshape(-1,self.embedding_size)
        Inter=self.interaction(Inter)
        Inter=Inter.reshape(batch_size,-1,self.num_classes)

        Inter=Inter.transpose(1,2)
        Inter=Inter.contiguous()
        Inter=Inter.reshape(batch_size*self.num_classes,-1)
        Inter=Inter.unsqueeze(1)

#        Agreg=F.dropout(Inter,0.5,training=self.training)

        Agreg=self.aggregation1(Inter)
        Agreg=F.relu(Agreg)
        Agreg=F.dropout(Agreg,0.5,training=self.training)
        Agreg=Agreg.reshape(batch_size*self.num_classes,1,-1)
        Agreg=self.aggregation2(Agreg)
        Agreg=Agreg.reshape(batch_size,self.num_classes)

        prob=F.softmax(Agreg,dim=1)
        class_=torch.max(prob,1)[1]
        
        return Agreg, prob, class_


class EXAM_ADDITIONAL_GRU(nn.Module):
    def __init__(self,voca_size,embedding_size,hidden_size,num_classes,max_length):
        super(EXAM_ADDITIONAL_GRU, self).__init__()

        # Parameter
        self.voca_size = voca_size
        self.embedding_size = embedding_size
        self.hidden_size=hidden_size
        self.num_classes = num_classes
        self.max_length = max_length
        
        
        # Word-level Encoder
        self.Embedding=nn.Embedding(self.voca_size,self.embedding_size)
        self.Embedding.weight.data.uniform_(-0.2,0.2)     

        self.GRU=nn.GRU(self.embedding_size,self.hidden_size,2)
        self.interaction = nn.Linear(self.hidden_size,self.num_classes,bias=False)
        self.interaction.weight.data.uniform_(-0.1,0.1)     
        
        self.aggregation1 = nn.Linear(self.max_length,self.max_length*2)
        self.aggregation1.weight.data.uniform_(-0.1,0.1)
        torch.nn.init.xavier_uniform_(self.aggregation1.weight)

        self.aggregation2 = nn.Linear(self.max_length*2,1)
        self.aggregation2.weight.data.uniform_(-0.1,0.1)
        torch.nn.init.xavier_uniform_(self.aggregation2.weight)    


    def forward(self,train_input):        
        batch=train_input.size()[0] # batch size
        
        # forward (GRU)
        ques = self.Embedding(train_input)
        ques = self.GRU(ques)
        word_level = ques[0]

        inter=self.interaction(word_level)
        inter=inter.transpose(1,2)

        agreg=self.aggregation1(inter)
        agreg=F.relu(agreg)
        agreg=F.dropout(agreg,0.5,training=self.training)
        agreg=self.aggregation2(agreg)
        agreg=agreg.reshape(batch,self.num_classes)
        
        prob=F.softmax(agreg,dim=1)
        class_=torch.max(prob,1)[1]
        
        return agreg, prob, class_


class EXAM_ENCODER(nn.Module):
    def __init__(self,voca_size,embedding_size,region_size,num_classes,max_length):
        super(EXAM_ENCODER, self).__init__()

        # Parameter
        self.voca_size = voca_size
        self.embedding_size = embedding_size
        self.region_size = region_size
        self.num_classes = num_classes
        self.max_length = max_length
        
        self.region_radius = self.region_size//2
        self.max_length = max_length
        self.entire = max_length-2*self.region_radius
        
        # Word-level Encoder
        self.embedding_region = nn.Embedding(self.voca_size,self.region_size*self.embedding_size,padding_idx=0)
        self.embedding_region.weight.data.uniform_(-0.01,0.01)

        self.embedding_word = nn.Embedding(self.voca_size,self.embedding_size,padding_idx=0)
        self.embedding_word.weight.data.uniform_(-0.01,0.01)

        # No Interaction Layer
#        self.interaction = nn.Linear(self.embedding_size,self.num_classes,bias=False)
#        self.interaction.weight.data.uniform_(-0.01,0.01)

        # Aggreation Layer
        self.aggregation1 = nn.Linear(self.embedding_size,self.max_length*2)
        torch.nn.init.xavier_uniform_(self.aggregation1.weight)

        self.aggregation2 = nn.Linear(self.max_length*2,self.num_classes)
        torch.nn.init.xavier_uniform_(self.aggregation2.weight)        


    def forward(self,train_input):        
        batch_size=train_input.size()[0] # batch size
        length=train_input.size()[1]     # max_length

        
        # target word
        trimed_seq=train_input[:,self.region_radius:length-self.region_radius]                # [batch, length-region_size+1]
        K = self.embedding_region(trimed_seq)                                       # [batch, length-region_size+1, Region_size*embedding_size]
        K = K.reshape(batch_size,-1,self.region_size,self.embedding_size)                     # [batch, length-region_size+1, Region_size, embedding_size]

        # Region word
        E = torch.cat([train_input[:,nu-self.region_radius:nu-self.region_radius+self.region_size].unsqueeze(0)                        for nu in range(self.region_radius,length-self.region_radius)],dim=0)       # [length-region_size+1, batch_size, region_size]
    
        E = E.transpose(1,0)                                                         # [batch_size, length-region_size+1, region_size]
        E = E.transpose(2,1)                                                         # [batch_size, region_size, length-region_size+1]
        E = self.embedding_word(E)
        E = E.reshape(batch_size,self.entire,self.region_size,self.embedding_size)                  # [batch, length-region_size+1, region_size, embedding_size]


        NoInter=torch.max(E*K,dim=2)[0]
        NoInter=torch.sum(NoInter,dim=1)

        Agreg=F.dropout(NoInter,0.5,training=self.training)

        Agreg=self.aggregation1(NoInter)
        Agreg=F.relu(Agreg)
        Agreg=F.dropout(Agreg,0.5,training=self.training)
        Agreg=self.aggregation2(Agreg)

        prob=F.softmax(Agreg,dim=1)
        class_=torch.max(prob,1)[1]
        
        return Agreg, prob, class_


class EXAM_ADDITIONAL_ATTENTION(nn.Module):
    def __init__(self,voca_size,embedding_size,region_size,num_classes,max_length):
        super(EXAM_ADDITIONAL_ATTENTION, self).__init__()

        # Parameter
        self.voca_size = voca_size
        self.embedding_size = embedding_size
        self.region_size = region_size
        self.num_classes = num_classes
        self.max_length = max_length
        
        self.region_radius = self.region_size//2
        self.max_length = max_length
        self.entire = max_length-2*self.region_radius

        # Word-level Encoder
        self.embedding_region = nn.Embedding(self.voca_size,self.region_size*self.embedding_size,padding_idx=0)
        self.embedding_region.weight.data.uniform_(-0.01,0.01)

        self.embedding_word = nn.Embedding(self.voca_size,self.embedding_size,padding_idx=0)
        self.embedding_word.weight.data.uniform_(-0.01,0.01)

        # Interaction Layer
        self.interaction = nn.Linear(self.embedding_size,self.num_classes,bias=False)
        self.interaction.weight.data.uniform_(-0.01,0.01)

        # Aggreation Layer
        self.aggregation1 = nn.Linear(self.embedding_size,self.max_length*2)
        torch.nn.init.xavier_uniform_(self.aggregation1.weight)

        self.aggregation2 = nn.Linear(self.max_length*2,1)
        torch.nn.init.xavier_uniform_(self.aggregation2.weight)     


    def forward(self,train_input):        
        batch_size=train_input.size()[0] # batch size
        length=train_input.size()[1]     # max_length

        # target word
        trimed_seq=train_input[:,self.region_radius:length-self.region_radius]  # [batch, length-region_size+1]
        K = self.embedding_region(trimed_seq)                                   # [batch, length-region_size+1, Region_size*embedding_size]
        K = K.reshape(batch_size,-1,self.region_size,self.embedding_size)       # [batch, length-region_size+1, Region_size, embedding_size]

        
        # Region word
        E = torch.cat([train_input[:,nu-self.region_radius:nu-self.region_radius+self.region_size].unsqueeze(0)                    for nu in range(self.region_radius,length-self.region_radius)],dim=0) # [length-region_size+1, batch_size, region_size]

        
        E = E.transpose(1,0)                                          # [batch_size, length-region_size+1, region_size]
        E = E.transpose(2,1)                                          # [batch_size, region_size, length-region_size+1]
        E1 = self.embedding_word(E)
        E2 = E1.reshape(batch_size,self.entire,self.region_size,self.embedding_size) # [batch, length-region_size+1, region_size, embedding_size]


        Inter=torch.max(E2*K,dim=2)[0]                         # [batch, length-region_size+1,embedding_size]
        
        #Attention
        att=self.interaction(Inter.reshape(-1,self.embedding_size))    # [batch*(length-region_size+1),embedding_size]
        att=att.reshape(batch_size,-1,self.num_classes)         # [batch,length-region_size+1,num_classes]
        att= F.softmax(att,dim=1)
        Inter= torch.bmm(Inter.transpose(1,2),att)

        Inter=Inter.contiguous()
        Inter=Inter.reshape(batch_size*self.num_classes,-1)
        Inter=Inter.unsqueeze(1)

        #Aggregation
        Agreg=F.dropout(Inter,0.5,training=self.training)
        Agreg=self.aggregation1(Inter)
        Agreg=F.relu(Agreg)
        Agreg=F.dropout(Agreg,0.5,training=self.training)
        Agreg=Agreg.reshape(batch_size*self.num_classes,1,-1)
        Agreg=self.aggregation2(Agreg)
        Agreg=Agreg.reshape(batch_size,self.num_classes)

        prob=F.softmax(Agreg,dim=1)
        class_=torch.max(prob,1)[1]
        
        return Agreg, prob, class_


MODEL_SETTING_NAME = ['EXAM','EXAM_ADDITIONAL_GRU','EXAM_ENCODER','EXAM_ADDITIONAL_ATTENTION']

def EXAM_train(train,valid,test,EXAM_model,Route,word_dict,lr_rate=0.0001,epoches=20,batch_size=256,l2norm=1e-8):
    print("\n","\033[1m]"+"Data Summary")
    print(" - The number of train_data :",len(train))
    print(" - The number of valid_data :",len(valid))
    print(" - The number of test_data :",len(test))
    print(" - The number of word :",len(word_dict))


    print("\n","\033[1m]"+"Training EXplict interAction Model(EXAM) for text classification")


    print(" - Learning_rate :",lr_rate)
    print(" - Max Epoch :",epoches)
    print(" - Batch :",batch_size)
    print(" - Optimizer : Adam")
    print(" - L2norm :",l2norm)
    print(" - Dropout : On")
    print(" - Loss : Cross Entropy\n\n")

    voca_size=len(word_dict)
    embedding_size=128
    region_size=7
    num_classes=Route["num_classes"]
    max_length=Route["max_length"]
    
    
    model=EXAM_model(voca_size,embedding_size,region_size,num_classes,max_length).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    loss_function = nn.CrossEntropyLoss()                               # loss function : Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr = lr_rate, weight_decay=l2norm)      # optimizer :adam #지금하고있는건 없음

    for num in range(epoches*int(len(train)/batch_size)+1):
        time_test=time.time()
        model.train()                                                   # using dropout

        train_batch = random.sample(train,batch_size)
        train_batch=split_padding(train_batch,max_length,word_dict)
        train_input, train_label = zip(*train_batch)
        train_input=torch.tensor(train_input).to(device) # to(device)
        train_label=torch.tensor(train_label).to(device) # to(device)


        train_logits, train_probs, train_classes = model(train_input)   # training
        losses = loss_function(train_logits, train_label) # calculate loss

        optimizer.zero_grad()                                           # gradient to zero
        losses.backward()                                               # load backward function

        optimizer.step()                                                # update parameters
        if num==2:
            timer=(time.time()-time_test)*int(len(train)/batch_size)
            h=timer//3600
            m=timer//60-h*60
            s=timer-m*60-h*3600
            print(" - Expected Running Time of each epoch : %dh %dm %ds" % (int(h),int(m),int(s)),"\n")

        # validation
        if num % int(len(train)/batch_size) == 0:

                train_accuracy = torch.sum(torch.tensor([train_classes[i]==train_label[i] 
                                         for i in range(len(train_classes))]).to(device), dtype=torch.float)/batch_size


                model.eval()                                                # not useing dropout
                valid_input, valid_label = zip(*valid)


                valid_accuracy=[]
                for nv in range(int(len(valid_input)/200)):
                    valid_in=valid_input[nv*200:(nv+1)*200]
                    valid_la=valid_label[nv*200:(nv+1)*200]
                    valid_in=torch.tensor(valid_in).to(device)
                    valid_la=torch.tensor(valid_la).to(device)                

                    valid_logits, valid_probs, valid_classes = model(valid_in)
                    valid_ac = torch.sum(torch.tensor([valid_classes[i]==valid_la[i]
                                                             for i in range(len(valid_classes))]).to(device), dtype=torch.float)/len(valid_in)

                valid_accuracy.append(valid_ac)


                model.eval()                                                # not useing dropout
                test_input, test_label = zip(*test)

                test_accuracy=[]
                for nv in range(int(len(test_input)/200)):
                    test_in=test_input[nv*200:(nv+1)*200]
                    test_la=test_label[nv*200:(nv+1)*200]
                    test_in=torch.tensor(test_in).to(device) # to(device)
                    test_la=torch.tensor(test_la).to(device) # to(device)

                    test_logits, test_probs, test_classes = model(test_in)
                    test_ac = torch.sum(torch.tensor([test_classes[i]==test_la[i]
                                                             for i in range(len(test_classes))]).to(device), dtype=torch.float)/len(test_in)

                    test_accuracy.append(test_ac)

                print("Epoch :",num/int(len(train)/batch_size),
                      " Train_accuracy :",round(float(train_accuracy),4),
                      " Valid_accuracy :",round(float(torch.sum(torch.tensor(valid_accuracy).to(device))/(len(valid_accuracy))),4),
                      " Test_accuracy :",round(float(torch.sum(torch.tensor(test_accuracy).to(device))/(len(test_accuracy))),4))



        if (num > 0) & (num % 5000 == 0):

                train_accuracy = torch.sum(torch.tensor([train_classes[i]==train_label[i] 
                                         for i in range(len(train_classes))]), dtype=torch.float)/batch_size

                model.eval()                                                # not useing dropout

                valid_accuracy=[]
                for nv in range(int(len(valid_input)/200)):
                    valid_in=valid_input[nv*200:(nv+1)*200]
                    valid_la=valid_label[nv*200:(nv+1)*200]
                    valid_in=torch.tensor(valid_in).to(device)
                    valid_la=torch.tensor(valid_la).to(device)

                    valid_logits, valid_probs, valid_classes = model(valid_in)
                    valid_ac = torch.sum(torch.tensor([valid_classes[i]==valid_la[i]
                                                             for i in range(len(valid_classes))]).to(device), dtype=torch.float)/len(valid_in)

                valid_accuracy.append(valid_ac)

                model.eval()                                                # not useing dropout


                test_accuracy=[]
                for nv in range(int(len(test_input)/200)):
                    test_in=test_input[nv*200:(nv+1)*200]
                    test_la=test_label[nv*200:(nv+1)*200]
                    test_in=torch.tensor(test_in).to(device) # to(device)
                    test_la=torch.tensor(test_la).to(device) # to(device)

                    test_logits, test_probs, test_classes = model(test_in)
                    test_ac = torch.sum(torch.tensor([test_classes[i]==test_la[i]
                                                             for i in range(len(test_classes))]).to(device), dtype=torch.float)/len(test_in)

                    test_accuracy.append(test_ac)

                print("Epoch :",num/int(len(train)/batch_size),
                      " Train_accuracy :",round(float(train_accuracy),4),
                      " Valid_accuracy :",round(float(torch.sum(torch.tensor(valid_accuracy).to(device))/(len(valid_accuracy))),4),
                      " Test_accuracy :",round(float(torch.sum(torch.tensor(test_accuracy).to(device))/(len(test_accuracy))),4))

    
    return model    


# In[ ]:

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')

Name=["AG","DBP","Yah. A.","Amz F.","Amz P."]
Route=DataName[Name[2]]


train_raw, test_raw = read_data(Route["path"])
word_dict = dictionary(train_raw,Route["word_min"])
train, valid = train_valid(train_raw)

test  = split_padding(test_raw,Route["max_length"],word_dict)
valid = split_padding(valid,Route["max_length"],word_dict)

MODEL_SETTING_NAME = ['EXAM','EXAM_ADDITIONAL_GRU']
print("Please select model information -",MODEL_SETTING_NAME)
print("Please select l2norm candidate : 1e-5, 1e-7, 1e-8, 1e-9")


# In[ ]:

EXAM_train(train,valid,test,EXAM,Route,word_dict,epoches=5,l2norm=1e-8)

