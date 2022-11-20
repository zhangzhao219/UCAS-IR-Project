from tqdm import tqdm
from rich_progress import progress
from torch.utils.data import Dataset

class LoadTrainData(Dataset):
    def __init__(self,data,tokenizer):
        super(LoadTrainData,self).__init__()

        self.input_ids_positive = []
        self.token_type_ids_positive = []
        self.attention_mask_positive = []

        self.input_ids_negative = []
        self.token_type_ids_negative = []
        self.attention_mask_negative = []

        # for _, row in progress.track(data.iterrows(), total=data.shape[0], description="Load Train Data..."):
        for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Load Train Data"):
            bert_inputs_dict_positive = tokenizer(row[0], text_pair='{}'.format(row[1]), max_length=192, padding='max_length', truncation=True, return_tensors='pt')
            self.input_ids_positive.append(bert_inputs_dict_positive['input_ids'])
            self.token_type_ids_positive.append(bert_inputs_dict_positive['token_type_ids'])
            self.attention_mask_positive.append(bert_inputs_dict_positive['attention_mask'])

            bert_inputs_dict_negative = tokenizer(row[0], text_pair='{}'.format(row[2]), max_length=192, padding='max_length', truncation=True, return_tensors='pt')
            self.input_ids_negative.append(bert_inputs_dict_negative['input_ids'])
            self.token_type_ids_negative.append(bert_inputs_dict_negative['token_type_ids'])
            self.attention_mask_negative.append(bert_inputs_dict_negative['attention_mask'])

    def __getitem__(self,index):
        return (self.input_ids_positive[index], self.token_type_ids_positive[index], self.attention_mask_positive[index], self.input_ids_negative[index], self.token_type_ids_negative[index], self.attention_mask_negative[index])

    def __len__(self):
        return len(self.input_ids_positive)

class LoadTestData(Dataset):
    def __init__(self,data,tokenizer):
        super(LoadTestData,self).__init__()
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []

        # for _, row in progress.track(data.iterrows(), total=data.shape[0], description="Load Test Data..."):
        for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Load Test Data"):
            bert_inputs_dict = tokenizer(row[2], text_pair='{}'.format(row[3]), max_length=192, padding='max_length', truncation=True, return_tensors='pt')
            self.input_ids.append(bert_inputs_dict['input_ids'])
            self.token_type_ids.append(bert_inputs_dict['token_type_ids'])
            self.attention_mask.append(bert_inputs_dict['attention_mask'])

    def __getitem__(self,index):
        return (self.input_ids[index], self.token_type_ids[index], self.attention_mask[index])

    def __len__(self):
        return len(self.input_ids)


class LoadTrainData_Colbert(Dataset):
    def __init__(self,data,tokenizer):
        super(LoadTrainData_Colbert,self).__init__()

        self.query_input_ids = []
        self.query_token_type_ids = []
        self.query_attention_mask = []

        self.document_input_ids_positive = []
        self.document_token_type_ids_positive = []
        self.document_attention_mask_positive = []

        self.document_input_ids_negative = []
        self.document_token_type_ids_negative = []
        self.document_attention_mask_negative = []

        # for _, row in progress.track(data.iterrows(), total=data.shape[0], description="Load Train Data..."):
        for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Load Train Data"):

            query_bert_inputs_dict = tokenizer(row[0], max_length=192, padding='max_length', truncation=True, return_tensors='pt')
            self.query_input_ids.append(query_bert_inputs_dict['input_ids'])
            self.query_token_type_ids.append(query_bert_inputs_dict['token_type_ids'])
            self.query_attention_mask.append(query_bert_inputs_dict['attention_mask'])

            document_bert_inputs_dict_positive = tokenizer(row[1], max_length=192, padding='max_length', truncation=True, return_tensors='pt')
            self.document_input_ids_positive.append(document_bert_inputs_dict_positive['input_ids'])
            self.document_token_type_ids_positive.append(document_bert_inputs_dict_positive['token_type_ids'])
            self.document_attention_mask_positive.append(document_bert_inputs_dict_positive['attention_mask'])

            document_bert_inputs_dict_negative = tokenizer(row[2], max_length=192, padding='max_length', truncation=True, return_tensors='pt')
            self.document_input_ids_negative.append(document_bert_inputs_dict_negative['input_ids'])
            self.document_token_type_ids_negative.append(document_bert_inputs_dict_negative['token_type_ids'])
            self.document_attention_mask_negative.append(document_bert_inputs_dict_negative['attention_mask'])

    def __getitem__(self,index):
        return (
            self.query_input_ids[index], self.query_token_type_ids[index], self.query_attention_mask[index], 
            self.document_input_ids_positive[index], self.document_token_type_ids_positive[index], self.document_attention_mask_positive[index],
            self.document_input_ids_negative[index], self.document_token_type_ids_negative[index], self.document_attention_mask_negative[index]
        )

    def __len__(self):
        return len(self.query_input_ids)

class LoadTestData_Colbert(Dataset):
    def __init__(self,data,tokenizer):
        super(LoadTestData_Colbert,self).__init__()

        self.query_input_ids = []
        self.query_token_type_ids = []
        self.query_attention_mask = []

        self.document_input_ids = []
        self.document_token_type_ids = []
        self.document_attention_mask = []

        # for _, row in progress.track(data.iterrows(), total=data.shape[0], description="Load Test Data..."):
        for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Load Test Data"):

            query_bert_inputs_dict = tokenizer(row[2], max_length=192, padding='max_length', truncation=True, return_tensors='pt')
            self.query_input_ids.append(query_bert_inputs_dict['input_ids'])
            self.query_token_type_ids.append(query_bert_inputs_dict['token_type_ids'])
            self.query_attention_mask.append(query_bert_inputs_dict['attention_mask'])

            document_bert_inputs_dict = tokenizer(row[3], max_length=192, padding='max_length', truncation=True, return_tensors='pt')
            self.document_input_ids.append(document_bert_inputs_dict['input_ids'])
            self.document_token_type_ids.append(document_bert_inputs_dict['token_type_ids'])
            self.document_attention_mask.append(document_bert_inputs_dict['attention_mask'])

    def __getitem__(self,index):
        return (
            self.query_input_ids[index], self.query_token_type_ids[index], self.query_attention_mask[index],
            self.document_input_ids[index], self.document_token_type_ids[index], self.document_attention_mask[index],
        )

    def __len__(self):
        return len(self.query_input_ids)

