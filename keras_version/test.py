# for i in train_data:
        #     temp=i.split('\t')
        #     word_len=[len(s.split(' ')) for s in temp[0].split('<eou>')]
        #     word_len.append(len(temp[1].split(' ')))
        #     if len(temp)==3 and len(temp[0].split('<eou>'))<=self.max_sen and max(word_len)<=self.max_len:
        #         train_data_.append(i)
        #         words=temp[0].split(' ')+temp[1].split(' ')
        #         for j in words:
        #             self.dict.add(j)
        #             try:
        #                 expand_list=self.concept_data[j].keys()
        #                 for t in expand_list:
        #                     self.dict.add(t)
        #             except:
        #                 pass
        # for i in valid_data:
        #     temp=i.split('\t')
        #     word_len = [len(s.split(' ')) for s in temp[0].split('<eou>')]
        #     word_len.append(len(temp[1].split(' ')))
        #     if len(temp)==3 and len(temp[0].split('<eou>'))<=self.max_sen and max(word_len)<=self.max_len:
        #         valid_data_.append(i)
        #         words=temp[0].split(' ')+temp[1].split(' ')
        #         for j in words:
        #             self.dict.add(j)
        #             try:
        #                 expand_list = self.concept_data[j].keys()
        #                 for t in expand_list:
        #                     self.dict.add(t)
        #             except:
        #                 pass
        #
        # print(len(train_data),len(train_data_))
        # # with open('../Data/train_3.txt','w',encoding='utf-8') as f:
        # #     f.writelines(train_data_)
        # # with open('../Data/valid_3.txt','w',encoding='utf-8') as f:
        # #     f.writelines(valid_data_)
        # dict_index={w:i for i,w in enumerate(self.dict)}
        # with open('../Data/all_dict.json','w',encoding='utf-8') as f:
        #     json.dump(dict_index,f)