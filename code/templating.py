from arguments import get_args_parser
# from random_words import RandomWords



def get_temps(tokenizer):
    args = get_args_parser()
    temps = {}
    with open(args.data_dir + "/" + args.temps, "r") as f:
        for i in f.readlines():
            i = i.strip().split("\t")
            info = {}
            info['name'] = i[1].strip()
            info['temp'] = [
                    # ['the', tokenizer.mask_token],
                    # [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
                    # ['the', tokenizer.mask_token],

                # [tokenizer.mask_token],
                # [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
                # [ tokenizer.mask_token],

                [tokenizer.mask_token],
                [tokenizer.mask_token],
                [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],

            ]

            print (i)
            info['labels'] = [
                # (i[2],),
                # (i[3],i[4],i[5]),
                # (i[6],)

                (i[2],),
                (i[6],),
                (i[3], i[4], i[5]),

            ]
            print (info)
            temps[info['name']] = info
    return temps

def get_semeval_temps(tokenizer):
    # random_word = RandomWords()
    # words = random_word.random_words(count=8)
    args = get_args_parser()
    temps = {}
    with open(args.data_dir + "/" + args.temps, "r") as f:
        for i in f.readlines():
            i = i.strip().split("\t")
            info = {}
            info['name'] = i[1].strip()
            info['temp'] = [
                # ['the', tokenizer.mask_token, words[1], words[2]],
                # ['was', tokenizer.mask_token, words[4]],
                # ['the', tokenizer.mask_token, words[6], words[7]],

                # ['the', tokenizer.mask_token, 'was'],
                # [tokenizer.mask_token],
                # [tokenizer.mask_token],

                # ['the', tokenizer.mask_token],
                #
                # ['the', tokenizer.mask_token],
                # ['was', tokenizer.mask_token, 'to'],

                [tokenizer.mask_token],

                [tokenizer.mask_token],
                [tokenizer.mask_token,],

                # ['the', tokenizer.mask_token, 'the', 'entity'],
                # ['was', tokenizer.mask_token, 'to'],
                # ['the', tokenizer.mask_token, 'the', 'entity'],
             ]
            print(i)

            # info['labels'] = [
            #     (i[2],),
            #     (i[3],),
            #     (i[4],)
            # ]

            info['labels'] = [
                (i[2],),
                (i[4],),
                (i[3],),
            ]


            print (info)
            temps[info['name']] = info
    return temps

