import codecs

def count_raw_data(infile):
    readt = codecs.open(infile, 'r', encoding='utf-8')
    datat = readt.readlines()

    dict_title = dict()

    for item in datat:
        item_tmp = item.strip()
        #if len(item_tmp) > 7:
        if '<title>' in item_tmp:
            cur_title = item_tmp[7:]
            dict_title[cur_title] = 0
        else:
            dict_title[cur_title] += len(item_tmp)

    total_num = 0
    for item in dict_title:
        print (str(item) + ': ' + str(dict_title[item]) )
        total_num += dict_title[item]

    print ("The total num is " + str(total_num))

    #return dict_title

def check_ent_num(infile, type):
    readt = codecs.open(infile, 'r', encoding='utf-8')
    datat = readt.readlines()
    line_num = 1
    novel_vocab_all = dict()

    if type == "novel":
        for item in datat:
            line = item.strip()
            if line_num % 14 == 0:  # novel end
                novel_vocab_all[title] = entity_num
                entity_num = 0
            elif line_num % 14 == 1:  # title
                title = line
            elif (line_num % 14) in [3, 5, 7, 9, 11, 13]:
                entities = line.split()
                entity_num += len(entities)
            line_num += 1

    else:
        for item in datat:
            line = item.strip()
            if line_num % 3 == 0:  # annual report end
                novel_vocab_all[title] = entity_num
                entity_num = 0
            elif line_num % 3 == 1:  # title
                title = line
            else:
                entities = line.split()
                #print("The current entities are :")
                #print(len(entities))
                entity_num += len(entities)
                #print(len(entity_list))
            line_num += 1


    return novel_vocab_all

if __name__ == "__main__":
    count_raw_data('dataset_book9/book9_raw')


    count_raw_data('dataset_finance/annual_report_raw')


    count_raw_data('dataset_thuner/thuner_rawdata_small.txt')

