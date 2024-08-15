import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBartTokenizerFast
import torch.nn.functional as F
from transformers.models.mbart.modeling_mbart import shift_tokens_right
import os


class MBart(nn.Module):
    def __init__(self):
        super(MBart, self).__init__()
        self.mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

        self.encoder = self.mbart.get_encoder()
        self.decoder = self.mbart.get_decoder()

        self.lm_head = self.mbart.lm_head

        self.tokenizer = MBartTokenizerFast.from_pretrained("facebook/mbart-large-cc25")

        self.config = self.decoder.config

    def forward(self, text, src_lang='', tgt_lang=''):
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        print('tokenizer: ')
        encoded_input = self.tokenizer(text, return_tensors="pt",
                                       padding=True, truncation=True)

        print('encoded_input: ', encoded_input)
        decoded_input = self.tokenizer('I am a good student.', return_tensors="pt",
                                       padding=True, truncation=True)

        print('decoded_input ', decoded_input)

        # 编码输入文本
        input_ids = encoded_input.input_ids
        attention_mask = encoded_input.attention_mask

        decoder_input_ids = decoded_input.input_ids
        decoder_attention_mask = decoded_input.attention_mask

        # 使用encoder生成encoder隐藏状态
        encoder_outputs = self.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       return_dict=True)
        decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        print('shift_tokens_right: ', decoder_input_ids)
        # print('self.config.pad_token_id: ', self.config.pad_token_id)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,

            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        # 获取 logits
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        # 应用 Softmax 获取概率分布
        probabilities = F.softmax(logits, dim=-1)

        # 获取最大概率对应的 token IDs
        predicted_ids = torch.argmax(probabilities, dim=-1)

        print('forward: ', predicted_ids)


def dir_is_exist(base_path, subdir):
    dir_path = os.path.join(base_path, subdir)
    return os.path.exists(dir_path) and os.path.isdir(dir_path)


if __name__ == '__main__':
    # mbart = MBart()
    # # res = mbart.generate('I love you.', src_lang="en_XX", tgt_lang="es_XX")  # 设置目标语言为中文
    # mbart('I love you.', src_lang="en_XX", tgt_lang="en_XX")  # 设置目标语言为中文
    # mbart('I love you.', src_lang="zh_CN", tgt_lang="en_XX")  # 设置目标语言为中文
    # mbart('我爱你。', src_lang="zh_CN", tgt_lang="zh_CN")  # 设置目标语言为中文
    # mbart('我爱你。', src_lang="en_XX", tgt_lang="zh_CN")  # 设置目标语言为中文
    # # print(f"Translated text: {res}")
    # loss_lambda = torch.tensor('0.1')
    # print(loss_lambda)
    # input_path = '/Volumes/OneTouch/P14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test'
    # sub_dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    # print('sub_dirs: ', sub_dirs)
    # base_path = '/Volumes/OneTouch/P14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test/'
    # # base_path = ''
    # res = os.listdir(base_path)
    # print(res)
    #
    # if base_path:
    #     input_paths = [os.path.join(input_path, subdir) for subdir in sub_dirs if
    #                    not dir_is_exist(base_path, subdir)]
    #     print('1_input_paths: ', input_paths)
    # else:
    #     input_paths = [os.path.join(input_path, subdir) for subdir in sub_dirs]
    #     print('2_input_paths: ', input_paths)

    text = """内容已经存在，忽略： ./examples/train/23March_2011_Wednesday_tagesschau-8747
内容已经存在，忽略： ./examples/train/21November_2010_Sunday_tagesschau-5670
内容已经存在，忽略： ./examples/train/07December_2010_Tuesday_heute-5635
内容已经存在，忽略： ./examples/train/22April_2010_Thursday_tagesschau-3732
内容已经存在，忽略： ./examples/train/07June_2010_Monday_heute-2331
内容已经存在，忽略： ./examples/train/12February_2010_Friday_tagesschau-129
内容已经存在，忽略： ./examples/train/20November_2011_Sunday_tagesschau-538
内容已经存在，忽略： ./examples/train/27February_2010_Saturday_tagesschau-7768
内容已经存在，忽略： ./examples/train/16July_2011_Saturday_tagesschau-7529
内容已经存在，忽略： ./examples/train/01July_2009_Wednesday_tagesschau-4558
内容已经存在，忽略： ./examples/train/24October_2009_Saturday_tagesschau-4290
内容已经存在，忽略： ./examples/train/02December_2010_Thursday_tagesschau-3643
内容已经存在，忽略： ./examples/train/01October_2012_Monday_heute-447
内容已经存在，忽略： ./examples/train/23February_2010_Tuesday_heute-6241
内容已经存在，忽略： ./examples/train/15April_2010_Thursday_tagesschau-5779
内容已经存在，忽略： ./examples/train/28April_2010_Wednesday_heute-5182
内容已经存在，忽略： ./examples/train/29September_2012_Saturday_tagesschau-2968
内容已经存在，忽略： ./examples/train/10February_2010_Wednesday_tagesschau-2517
内容已经存在，忽略： ./examples/train/27November_2009_Friday_tagesschau-7336
内容已经存在，忽略： ./examples/train/02December_2009_Wednesday_tagesschau-4041
内容已经存在，忽略： ./examples/train/07June_2010_Monday_heute-2329
内容已经存在，忽略： ./examples/train/27February_2011_Sunday_tagesschau-6113
内容已经存在，忽略： ./examples/train/07October_2010_Thursday_tagesschau-4130
内容已经存在，忽略： ./examples/train/06March_2011_Sunday_tagesschau-3893
内容已经存在，忽略： ./examples/train/30January_2010_Saturday_tagesschau-1697
内容已经存在，忽略： ./examples/train/09May_2011_Monday_tagesschau-8580
内容已经存在，忽略： ./examples/train/17February_2011_Thursday_heute-392
内容已经存在，忽略： ./examples/train/22January_2010_Friday_tagesschau-909
内容已经存在，忽略： ./examples/train/01October_2012_Monday_tagesschau-5367
内容已经存在，忽略： ./examples/train/28February_2011_Monday_tagesschau-4981
内容已经存在，忽略： ./examples/train/24February_2011_Thursday_tagesschau-4764
内容已经存在，忽略： ./examples/train/27August_2009_Thursday_tagesschau-3282
内容已经存在，忽略： ./examples/train/17February_2011_Thursday_heute-390
内容已经存在，忽略： ./examples/train/19October_2010_Tuesday_heute-8053
内容已经存在，忽略： ./examples/train/02December_2011_Friday_tagesschau-8016
内容已经存在，忽略： ./examples/train/06September_2009_Sunday_tagesschau-5304
内容已经存在，忽略： ./examples/train/27November_2011_Sunday_tagesschau-5142
内容已经存在，忽略： ./examples/train/09September_2010_Thursday_tagesschau-4369
内容已经存在，忽略： ./examples/train/02October_2009_Friday_tagesschau-1364
内容已经存在，忽略： ./examples/train/03August_2010_Tuesday_heute-7611
内容已经存在，忽略： ./examples/train/14April_2010_Wednesday_tagesschau-3812
内容已经存在，忽略： ./examples/train/16November_2009_Monday_tagesschau-3100
内容已经存在，忽略： ./examples/train/04March_2011_Friday_tagesschau-2844
内容已经存在，忽略： ./examples/train/02October_2010_Saturday_tagesschau-1305
内容已经存在，忽略： ./examples/train/10August_2010_Tuesday_heute-1456
内容已经存在，忽略： ./examples/train/23September_2010_Thursday_heute-6471
内容已经存在，忽略： ./examples/train/08April_2010_Thursday_heute-4000
内容已经存在，忽略： ./examples/train/15February_2011_Tuesday_heute-7909
内容已经存在，忽略： ./examples/train/11December_2009_Friday_tagesschau-3506
内容已经存在，忽略： ./examples/train/20May_2010_Thursday_tagesschau-3159
内容已经存在，忽略： ./examples/train/20April_2011_Wednesday_tagesschau-3088
内容已经存在，忽略： ./examples/train/16March_2011_Wednesday_tagesschau-2219
内容已经存在，忽略： ./examples/train/09June_2010_Wednesday_heute-8265
内容已经存在，忽略： ./examples/train/06February_2011_Sunday_tagesschau-6915
内容已经存在，忽略： ./examples/train/08November_2010_Monday_heute-6880
内容已经存在，忽略： ./examples/train/23July_2009_Thursday_tagesschau-6756
内容已经存在，忽略： ./examples/train/12December_2009_Saturday_tagesschau-4737
内容已经存在，忽略： ./examples/train/04October_2010_Monday_tagesschau-4725
内容已经存在，忽略： ./examples/train/28September_2009_Monday_tagesschau-4675
内容已经存在，忽略： ./examples/train/22September_2010_Wednesday_tagesschau-3259
内容已经存在，忽略： ./examples/train/05February_2010_Friday_tagesschau-2913
内容已经存在，忽略： ./examples/train/30January_2010_Saturday_tagesschau-1690
内容已经存在，忽略： ./examples/train/01August_2011_Monday_heute-4871
内容已经存在，忽略： ./examples/train/11August_2009_Tuesday_tagesschau-4362
内容已经存在，忽略： ./examples/train/10April_2011_Sunday_tagesschau-1529
内容已经存在，忽略： ./examples/train/15July_2011_Friday_tagesschau-8718
内容已经存在，忽略： ./examples/train/18April_2010_Sunday_tagesschau-6654
内容已经存在，忽略： ./examples/train/14August_2011_Sunday_tagesschau-3239
内容已经存在，忽略： ./examples/train/07February_2010_Sunday_tagesschau-490
内容已经存在，忽略： ./examples/train/19May_2010_Wednesday_heute-1110
内容已经存在，忽略： ./examples/train/10December_2009_Thursday_tagesschau-7485
内容已经存在，忽略： ./examples/train/14July_2011_Thursday_heute-5866
内容已经存在，忽略： ./examples/train/17February_2010_Wednesday_tagesschau-4932
内容已经存在，忽略： ./examples/train/11August_2011_Thursday_heute-3190
内容已经存在，忽略： ./examples/train/18October_2010_Monday_heute-621
内容已经存在，忽略： ./examples/train/29May_2011_Sunday_tagesschau-765
内容已经存在，忽略： ./examples/train/01June_2011_Wednesday_heute-954
内容已经存在，忽略： ./examples/train/20March_2010_Saturday_heute-7293
内容已经存在，忽略： ./examples/train/28January_2010_Thursday_tagesschau-2815
内容已经存在，忽略： ./examples/train/22April_2010_Thursday_heute-2115
内容已经存在，忽略： ./examples/train/12February_2010_Friday_tagesschau-123
内容已经存在，忽略： ./examples/train/02August_2010_Monday_heute-1247
内容已经存在，忽略： ./examples/train/17February_2010_Wednesday_heute-1483
内容已经存在，忽略： ./examples/train/07October_2010_Thursday_heute-8562
内容已经存在，忽略： ./examples/train/15February_2011_Tuesday_tagesschau-7481
内容已经存在，忽略： ./examples/train/25November_2010_Thursday_tagesschau-2538
内容已经存在，忽略： ./examples/train/14June_2010_Monday_heute-1606
内容已经存在，忽略： ./examples/train/10April_2011_Sunday_tagesschau-1526
内容已经存在，忽略： ./examples/train/14June_2010_Monday_heute-1599
内容已经存在，忽略： ./examples/train/30July_2010_Friday_tagesschau-1512
内容已经存在，忽略： ./examples/train/16July_2009_Thursday_tagesschau-5059
内容已经存在，忽略： ./examples/train/30June_2009_Tuesday_tagesschau-2955
内容已经存在，忽略： ./examples/train/20September_2010_Monday_heute-2940
内容已经存在，忽略： ./examples/train/28August_2010_Saturday_tagesschau-8188
内容已经存在，忽略： ./examples/train/24January_2013_Thursday_heute-7268
内容已经存在，忽略： ./examples/train/12September_2009_Saturday_tagesschau-6601
内容已经存在，忽略： ./examples/train/14September_2010_Tuesday_tagesschau-5219
内容已经存在，忽略： ./examples/train/28April_2010_Wednesday_heute-5196
内容已经存在，忽略： ./examples/train/15July_2010_Thursday_tagesschau-4060
内容已经存在，忽略： ./examples/train/23May_2011_Monday_heute-6455
内容已经存在，忽略： ./examples/train/30September_2012_Sunday_tagesschau-4030
内容已经存在，忽略： ./examples/train/27February_2010_Saturday_tagesschau-7776
内容已经存在，忽略： ./examples/train/23November_2011_Wednesday_heute-2406
内容已经存在，忽略： ./examples/train/29November_2011_Tuesday_tagesschau-5614
内容已经存在，忽略： ./examples/train/22November_2011_Tuesday_heute-4652
内容已经存在，忽略： ./examples/train/11December_2009_Friday_tagesschau-3507
内容已经存在，忽略： ./examples/train/24August_2010_Tuesday_heute-3019
内容已经存在，忽略： ./examples/train/05October_2010_Tuesday_heute-1220
内容已经存在，忽略： ./examples/train/25March_2011_Friday_tagesschau-5822
内容已经存在，忽略： ./examples/train/07February_2011_Monday_heute-4670
内容已经存在，忽略： ./examples/train/20May_2010_Thursday_tagesschau-3155
内容已经存在，忽略： ./examples/train/31May_2011_Tuesday_heute-1900
内容已经存在，忽略： ./examples/train/09March_2011_Wednesday_heute-1747
内容已经存在，忽略： ./examples/train/06September_2010_Monday_tagesschau-1145
内容已经存在，忽略： ./examples/train/08February_2010_Monday_heute-1505
内容已经存在，忽略： ./examples/train/28July_2011_Thursday_tagesschau-8522
内容已经存在，忽略： ./examples/train/02December_2010_Thursday_heute-8452
内容已经存在，忽略： ./examples/train/25January_2010_Monday_tagesschau-8302
内容已经存在，忽略： ./examples/train/06February_2010_Saturday_tagesschau-7367
内容已经存在，忽略： ./examples/train/20March_2010_Saturday_heute-7281
内容已经存在，忽略： ./examples/train/15December_2010_Wednesday_heute-6018
内容已经存在，忽略： ./examples/train/23April_2010_Friday_tagesschau-3671
内容已经存在，忽略： ./examples/train/18November_2011_Friday_tagesschau-2146
内容已经存在，忽略： ./examples/train/07December_2011_Wednesday_heute-7804
内容已经存在，忽略： ./examples/train/22August_2010_Sunday_tagesschau-1349
内容已经存在，忽略： ./examples/train/30July_2011_Saturday_tagesschau-8587
内容已经存在，忽略： ./examples/train/08July_2011_Friday_tagesschau-6732
内容已经存在，忽略： ./examples/train/18July_2010_Sunday_tagesschau-6220
内容已经存在，忽略： ./examples/train/26September_2009_Saturday_tagesschau-2346
内容已经存在，忽略： ./examples/train/15February_2010_Monday_tagesschau-1838
内容已经存在，忽略： ./examples/train/26January_2010_Tuesday_heute-91
内容已经存在，忽略： ./examples/train/14August_2009_Friday_tagesschau-76
内容已经存在，忽略： ./examples/train/25January_2010_Monday_heute-1466
内容已经存在，忽略： ./examples/train/30July_2010_Friday_tagesschau-1519
内容已经存在，忽略： ./examples/train/03September_2010_Friday_tagesschau-8426
内容已经存在，忽略： ./examples/train/09August_2011_Tuesday_heute-2645
内容已经存在，忽略： ./examples/train/03February_2011_Thursday_heute-605
内容已经存在，忽略： ./examples/train/02August_2010_Monday_heute-1248
内容已经存在，忽略： ./examples/train/27November_2011_Sunday_tagesschau-5145
内容已经存在，忽略： ./examples/train/18May_2010_Tuesday_heute-3368
内容已经存在，忽略： ./examples/train/01November_2010_Monday_tagesschau-135
内容已经存在，忽略： ./examples/train/23October_2010_Saturday_tagesschau-899
内容已经存在，忽略： ./examples/train/29November_2009_Sunday_tagesschau-7052
内容已经存在，忽略： ./examples/train/19March_2011_Saturday_tagesschau-4691
内容已经存在，忽略： ./examples/train/12July_2010_Monday_heute-250
内容已经存在，忽略： ./examples/train/08July_2010_Thursday_tagesschau-5387
内容已经存在，忽略： ./examples/train/20July_2010_Tuesday_heute-5259
内容已经存在，忽略： ./examples/train/04December_2010_Saturday_tagesschau-4836
内容已经存在，忽略： ./examples/train/10January_2010_Sunday_tagesschau-483
内容已经存在，忽略： ./examples/train/29July_2010_Thursday_heute-7746
内容已经存在，忽略： ./examples/train/14April_2010_Wednesday_tagesschau-3810
内容已经存在，忽略： ./examples/train/01October_2012_Monday_heute-445
内容已经存在，忽略： ./examples/train/08April_2010_Thursday_tagesschau-3954
内容已经存在，忽略： ./examples/train/05January_2011_Wednesday_heute-3538
内容已经存在，忽略： ./examples/train/27August_2009_Thursday_tagesschau-3273
内容已经存在，忽略： ./examples/train/10January_2010_Sunday_tagesschau-484
内容已经存在，忽略： ./examples/train/12April_2011_Tuesday_tagesschau-684
内容已经存在，忽略： ./examples/train/10August_2010_Tuesday_heute-1445
内容已经存在，忽略： ./examples/train/06September_2009_Sunday_tagesschau-5303
内容已经存在，忽略： ./examples/train/02October_2012_Tuesday_heute-3493
内容已经存在，忽略： ./examples/train/17September_2010_Friday_tagesschau-3422
内容已经存在，忽略： ./examples/train/05November_2010_Friday_tagesschau-1874
内容已经存在，忽略： ./examples/train/31March_2010_Wednesday_tagesschau-1005
内容已经存在，忽略： ./examples/train/18May_2010_Tuesday_heute-3365
内容已经存在，忽略： ./examples/train/20February_2010_Saturday_tagesschau-7901
内容已经存在，忽略： ./examples/train/18February_2010_Thursday_tagesschau-4523
内容已经存在，忽略： ./examples/train/17April_2011_Sunday_tagesschau-2155
内容已经存在，忽略： ./examples/train/29September_2010_Wednesday_tagesschau-1784
内容已经存在，忽略： ./examples/train/26October_2009_Monday_tagesschau-1393
内容已经存在，忽略： ./examples/train/29September_2009_Tuesday_tagesschau-8533
内容已经存在，忽略： ./examples/train/25August_2009_Tuesday_heute-3305
内容已经存在，忽略： ./examples/train/05January_2011_Wednesday_tagesschau-2254
内容已经存在，忽略： ./examples/train/15February_2010_Monday_tagesschau-1836
内容已经存在，忽略： ./examples/train/11August_2010_Wednesday_tagesschau-5
内容已经存在，忽略： ./examples/train/27April_2010_Tuesday_heute-1029
内容已经存在，忽略： ./examples/train/28May_2011_Saturday_tagesschau-2599
内容已经存在，忽略： ./examples/train/13April_2010_Tuesday_tagesschau-2310
内容已经存在，忽略： ./examples/train/21August_2010_Saturday_tagesschau-8820
内容已经存在，忽略： ./examples/train/14September_2010_Tuesday_heute-8241
内容已经存在，忽略： ./examples/train/09December_2010_Thursday_heute-7557
内容已经存在，忽略： ./examples/train/29November_2009_Sunday_tagesschau-7040
内容已经存在，忽略： ./examples/train/22November_2011_Tuesday_tagesschau-3648
内容已经存在，忽略： ./examples/train/01May_2010_Saturday_tagesschau-7186
内容已经存在，忽略： ./examples/train/19January_2011_Wednesday_tagesschau-7076
内容已经存在，忽略： ./examples/train/28April_2010_Wednesday_heute-5187
内容已经存在，忽略： ./examples/train/19March_2011_Saturday_tagesschau-4689
内容已经存在，忽略： ./examples/train/25February_2010_Thursday_tagesschau-1760
内容已经存在，忽略： ./examples/train/26October_2009_Monday_tagesschau-1398
内容已经存在，忽略： ./examples/train/22July_2010_Thursday_heute-8813
内容已经存在，忽略： ./examples/train/02December_2010_Thursday_heute-8446
内容已经存在，忽略： ./examples/train/24November_2011_Thursday_tagesschau-8334
内容已经存在，忽略： ./examples/train/02June_2010_Wednesday_heute-6791
内容已经存在，忽略： ./examples/train/20November_2009_Friday_tagesschau-6564
内容已经存在，忽略： ./examples/train/09August_2011_Tuesday_heute-2647
内容已经存在，忽略： ./examples/train/14April_2010_Wednesday_heute-1885
内容已经存在，忽略： ./examples/train/10August_2010_Tuesday_heute-1446
内容已经存在，忽略： ./examples/train/23August_2010_Monday_heute-5323
内容已经存在，忽略： ./examples/train/13March_2011_Sunday_tagesschau-4459
内容已经存在，忽略： ./examples/train/23February_2011_Wednesday_heute-3775
内容已经存在，忽略： ./examples/train/24August_2009_Monday_heute-6647
内容已经存在，忽略： ./examples/train/10August_2010_Tuesday_tagesschau-3869
内容已经存在，忽略： ./examples/train/01June_2011_Wednesday_heute-952
内容已经存在，忽略： ./examples/train/23December_2010_Thursday_tagesschau-1433
内容已经存在，忽略： ./examples/train/25January_2010_Monday_heute-1468
内容已经存在，忽略： ./examples/train/22July_2010_Thursday_heute-8809
内容已经存在，忽略： ./examples/train/13May_2010_Thursday_tagesschau-7922
内容已经存在，忽略： ./examples/train/29July_2010_Thursday_heute-7745
内容已经存在，忽略： ./examples/train/01April_2010_Thursday_heute-6702
内容已经存在，忽略： ./examples/train/08July_2011_Friday_tagesschau-6733
内容已经存在，忽略： ./examples/train/11November_2010_Thursday_tagesschau-3569
内容已经存在，忽略： ./examples/train/13October_2009_Tuesday_tagesschau-1652
内容已经存在，忽略： ./examples/train/28November_2011_Monday_heute-1590
内容已经存在，忽略： ./examples/train/28December_2011_Wednesday_tagesschau-1907
内容已经存在，忽略： ./examples/train/12February_2010_Friday_tagesschau-130
内容已经存在，忽略： ./examples/train/12July_2010_Monday_heute-259
内容已经存在，忽略： ./examples/train/29March_2010_Monday_tagesschau-8392
内容已经存在，忽略： ./examples/train/13July_2011_Wednesday_tagesschau-7985
内容已经存在，忽略： ./examples/train/24March_2011_Thursday_heute-6374
内容已经存在，忽略： ./examples/train/06December_2010_Monday_heute-6201
内容已经存在，忽略： ./examples/train/19February_2010_Friday_tagesschau-4094
内容已经存在，忽略： ./examples/train/14February_2010_Sunday_tagesschau-3837
内容已经存在，忽略： ./examples/train/15September_2010_Wednesday_heute-2473
内容已经存在，忽略： ./examples/train/23November_2011_Wednesday_heute-2403
内容已经存在，忽略： ./examples/train/12October_2009_Monday_tagesschau-552
内容已经存在，忽略： ./examples/train/26August_2009_Wednesday_heute-6556
内容已经存在，忽略： ./examples/train/14October_2010_Thursday_heute-2189
内容已经存在，忽略： ./examples/train/01November_2010_Monday_heute-1708
内容已经存在，忽略： ./examples/train/20October_2011_Thursday_tagesschau-504
内容已经存在，忽略： ./examples/train/22January_2010_Friday_tagesschau-908
内容已经存在，忽略： ./examples/train/15February_2011_Tuesday_heute-7903
内容已经存在，忽略： ./examples/train/13January_2010_Wednesday_tagesschau-3828
内容已经存在，忽略： ./examples/train/24August_2010_Tuesday_heute-3009
内容已经存在，忽略： ./examples/train/15November_2009_Sunday_tagesschau-1423
内容已经存在，忽略： ./examples/train/12September_2010_Sunday_tagesschau-3913
内容已经存在，忽略： ./examples/train/01December_2011_Thursday_tagesschau-3472
内容已经存在，忽略： ./examples/train/20November_2011_Sunday_tagesschau-547
内容已经存在，忽略： ./examples/train/06October_2011_Thursday_tagesschau-824
内容已经存在，忽略： ./examples/train/06December_2011_Tuesday_heute-1545
内容已经存在，忽略： ./examples/train/12January_2011_Wednesday_heute-6392
内容已经存在，忽略： ./examples/train/23November_2011_Wednesday_heute-2401
内容已经存在，忽略： ./examples/train/24November_2011_Thursday_heute-226
内容已经存在，忽略： ./examples/train/04July_2010_Sunday_tagesschau-7207
内容已经存在，忽略： ./examples/train/24July_2011_Sunday_tagesschau-6625
内容已经存在，忽略： ./examples/train/24May_2011_Tuesday_heute-6061
内容已经存在，忽略： ./examples/train/30August_2011_Tuesday_heute-792
内容已经存在，忽略： ./examples/train/29October_2009_Thursday_tagesschau-1016
内容已经存在，忽略： ./examples/train/10January_2011_Monday_heute-1367
内容已经存在，忽略： ./examples/train/16June_2010_Wednesday_heute-8327
内容已经存在，忽略： ./examples/train/29June_2011_Wednesday_tagesschau-4483
内容已经存在，忽略： ./examples/train/13July_2010_Tuesday_tagesschau-4312
内容已经存在，忽略： ./examples/train/20October_2009_Tuesday_tagesschau-3596
内容已经存在，忽略： ./examples/train/11August_2011_Thursday_heute-3191
内容已经存在，忽略： ./examples/train/13December_2010_Monday_tagesschau-6936
内容已经存在，忽略： ./examples/train/20November_2009_Friday_tagesschau-6570
内容已经存在，忽略： ./examples/train/30May_2010_Sunday_tagesschau-4159
内容已经存在，忽略： ./examples/train/21July_2009_Tuesday_tagesschau-8063
内容已经存在，忽略： ./examples/train/02June_2010_Wednesday_heute-6794
内容已经存在，忽略： ./examples/train/05December_2011_Monday_tagesschau-4172
内容已经存在，忽略： ./examples/train/13January_2010_Wednesday_tagesschau-3824
内容已经存在，忽略： ./examples/train/02September_2010_Thursday_heute-3458
内容已经存在，忽略： ./examples/train/15February_2010_Monday_tagesschau-1842
内容已经存在，忽略： ./examples/train/20January_2010_Wednesday_heute-814
内容已经存在，忽略： ./examples/train/28November_2011_Monday_heute-1593
内容已经存在，忽略： ./examples/train/02June_2010_Wednesday_heute-6793
内容已经存在，忽略： ./examples/train/07August_2009_Friday_tagesschau-6171
内容已经存在，忽略： ./examples/train/25March_2011_Friday_tagesschau-5827
内容已经存在，忽略： ./examples/train/06February_2010_Saturday_tagesschau-7361
内容已经存在，忽略： ./examples/train/05December_2009_Saturday_tagesschau-4700
内容已经存在，忽略： ./examples/train/26August_2009_Wednesday_tagesschau-3216
内容已经存在，忽略： ./examples/train/07June_2010_Monday_heute-2320
内容已经存在，忽略： ./examples/train/12August_2009_Wednesday_tagesschau-1803
内容已经存在，忽略： ./examples/train/27February_2011_Sunday_tagesschau-6116
内容已经存在，忽略： ./examples/train/28April_2010_Wednesday_heute-5198
内容已经存在，忽略： ./examples/train/31August_2010_Tuesday_tagesschau-5169
内容已经存在，忽略： ./examples/train/05May_2010_Wednesday_tagesschau-3350
内容已经存在，忽略： ./examples/train/03November_2009_Tuesday_tagesschau-8140
内容已经存在，忽略： ./examples/train/31August_2010_Tuesday_heute-3181
内容已经存在，忽略： ./examples/train/28January_2010_Thursday_tagesschau-2812
内容已经存在，忽略： ./examples/train/22September_2010_Wednesday_heute-2623
内容已经存在，忽略： ./examples/train/25May_2010_Tuesday_tagesschau-1298
内容已经存在，忽略： ./examples/train/09July_2009_Thursday_tagesschau-7118
内容已经存在，忽略： ./examples/train/30May_2011_Monday_heute-3673
内容已经存在，忽略： ./examples/train/11August_2010_Wednesday_tagesschau-4
内容已经存在，忽略： ./examples/train/29May_2010_Saturday_tagesschau-8416
内容已经存在，忽略： ./examples/train/23July_2009_Thursday_tagesschau-6757
内容已经存在，忽略： ./examples/train/24February_2011_Thursday_heute-6670
内容已经存在，忽略： ./examples/train/27November_2011_Sunday_tagesschau-5151
内容已经存在，忽略： ./examples/train/21April_2010_Wednesday_heute-3116
内容已经存在，忽略： ./examples/train/23January_2011_Sunday_tagesschau-7830
内容已经存在，忽略： ./examples/train/23November_2011_Wednesday_heute-2396
内容已经存在，忽略： ./examples/train/25February_2011_Friday_tagesschau-567
内容已经存在，忽略： ./examples/train/16June_2010_Wednesday_heute-8319
内容已经存在，忽略： ./examples/train/26May_2010_Wednesday_heute-7875
内容已经存在，忽略： ./examples/train/16May_2010_Sunday_tagesschau-5234
内容已经存在，忽略： ./examples/train/01October_2010_Friday_tagesschau-2693
内容已经存在，忽略： ./examples/train/28October_2010_Thursday_heute-337
内容已经存在，忽略： ./examples/train/17April_2010_Saturday_tagesschau-397
内容已经存在，忽略： ./examples/train/29May_2010_Saturday_tagesschau-8412
内容已经存在，忽略： ./examples/train/18January_2011_Tuesday_tagesschau-6998
内容已经存在，忽略： ./examples/train/27July_2010_Tuesday_heute-5266
内容已经存在，忽略： ./examples/train/22June_2011_Wednesday_heute-4385
内容已经存在，忽略： ./examples/train/28January_2013_Monday_tagesschau-3627
内容已经存在，忽略： ./examples/train/28January_2010_Thursday_heute-2898
内容已经存在，忽略： ./examples/train/05January_2011_Wednesday_tagesschau-2252
内容已经存在，忽略： ./examples/train/07April_2010_Wednesday_heute-1622
内容已经存在，忽略： ./examples/train/07June_2010_Monday_tagesschau-1558
内容已经存在，忽略： ./examples/train/25July_2011_Monday_heute-8227
内容已经存在，忽略： ./examples/train/14July_2010_Wednesday_heute-6181
内容已经存在，忽略： ./examples/train/24August_2010_Tuesday_tagesschau-2201
内容已经存在，忽略： ./examples/train/14August_2009_Friday_tagesschau-69
内容已经存在，忽略： ./examples/train/28October_2010_Thursday_heute-336
内容已经存在，忽略： ./examples/train/04April_2011_Monday_heute-971
内容已经存在，忽略： ./examples/train/12December_2011_Monday_heute-5802
内容已经存在，忽略： ./examples/train/22May_2010_Saturday_tagesschau-2739
内容已经存在，忽略： ./examples/train/04January_2010_Monday_heute-7406
内容已经存在，忽略： ./examples/train/21January_2011_Friday_tagesschau-4892
内容已经存在，忽略： ./examples/train/03February_2010_Wednesday_tagesschau-2059
内容已经存在，忽略： ./examples/train/11January_2011_Tuesday_heute-8180
内容已经存在，忽略： ./examples/train/24January_2013_Thursday_heute-7257
内容已经存在，忽略： ./examples/train/04July_2009_Saturday_tagesschau-3341
内容已经存在，忽略： ./examples/train/30September_2009_Wednesday_tagesschau-1780
内容已经存在，忽略： ./examples/train/09June_2010_Wednesday_heute-8256
内容已经存在，忽略： ./examples/train/14July_2011_Thursday_heute-5860
内容已经存在，忽略： ./examples/train/09July_2010_Friday_tagesschau-593
内容已经存在，忽略： ./examples/train/22February_2011_Tuesday_tagesschau-8404
内容已经存在，忽略： ./examples/train/01April_2010_Thursday_heute-6701
内容已经存在，忽略： ./examples/train/06May_2011_Friday_tagesschau-6433
内容已经存在，忽略： ./examples/train/24July_2010_Saturday_tagesschau-6337
内容已经存在，忽略： ./examples/train/07April_2010_Wednesday_tagesschau-6028
内容已经存在，忽略： ./examples/train/10March_2011_Thursday_heute-60
内容已经存在，忽略： ./examples/train/20October_2011_Thursday_tagesschau-505
内容已经存在，忽略： ./examples/train/27June_2010_Sunday_tagesschau-7388
内容已经存在，忽略： ./examples/train/26January_2013_Saturday_tagesschau-7343
内容已经存在，忽略： ./examples/train/09January_2010_Saturday_tagesschau-1925
内容已经存在，忽略： ./examples/train/07October_2010_Thursday_heute-8570
内容已经存在，忽略： ./examples/train/09February_2011_Wednesday_heute-7537
内容已经存在，忽略： ./examples/train/03December_2009_Thursday_tagesschau-7221
内容已经存在，忽略： ./examples/train/15February_2010_Monday_tagesschau-1837
内容已经存在，忽略： ./examples/train/19October_2009_Monday_tagesschau-236
内容已经存在，忽略： ./examples/train/04April_2011_Monday_heute-963
内容已经存在，忽略： ./examples/train/02December_2010_Thursday_tagesschau-3635
内容已经存在，忽略： ./examples/train/24October_2010_Sunday_tagesschau-2903
内容已经存在，忽略： ./examples/train/08February_2010_Monday_tagesschau-2802
内容已经存在，忽略： ./examples/train/23November_2011_Wednesday_heute-2395
内容已经存在，忽略： ./examples/train/02February_2010_Tuesday_tagesschau-7954
内容已经存在，忽略： ./examples/train/11December_2009_Friday_tagesschau-3510
内容已经存在，忽略： ./examples/train/31August_2010_Tuesday_heute-3177
内容已经存在，忽略： ./examples/train/07June_2010_Monday_heute-2326
内容已经存在，忽略： ./examples/train/24August_2010_Tuesday_tagesschau-2199
内容已经存在，忽略： ./examples/train/16February_2010_Tuesday_tagesschau-6096
内容已经存在，忽略： ./examples/train/08April_2010_Thursday_heute-4003
内容已经存在，忽略： ./examples/train/19November_2011_Saturday_tagesschau-3695
内容已经存在，忽略： ./examples/train/03February_2010_Wednesday_tagesschau-2055
内容已经存在，忽略： ./examples/train/05October_2011_Wednesday_heute-1077
内容已经存在，忽略： ./examples/train/28January_2013_Monday_heute-7995
内容已经存在，忽略： ./examples/train/02February_2010_Tuesday_tagesschau-7959
内容已经存在，忽略： ./examples/train/12August_2009_Wednesday_tagesschau-1802
内容已经存在，忽略： ./examples/train/06November_2010_Saturday_tagesschau-1154
内容已经存在，忽略： ./examples/train/01September_2010_Wednesday_tagesschau-5040
内容已经存在，忽略： ./examples/train/12December_2009_Saturday_tagesschau-4734
内容已经存在，忽略： ./examples/train/10February_2010_Wednesday_tagesschau-2518
内容已经存在，忽略： ./examples/train/08June_2010_Tuesday_heute-1993
内容已经存在，忽略： ./examples/train/03February_2011_Thursday_heute-610
内容已经存在，忽略： ./examples/train/02August_2010_Monday_heute-1246
内容已经存在，忽略： ./examples/train/23February_2010_Tuesday_heute-6246
内容已经存在，忽略： ./examples/train/24September_2010_Friday_tagesschau-213
内容已经存在，忽略： ./examples/train/09June_2010_Wednesday_heute-8264
内容已经存在，忽略： ./examples/train/23February_2010_Tuesday_heute-6242
内容已经存在，忽略： ./examples/train/19April_2011_Tuesday_heute-5333
内容已经存在，忽略： ./examples/train/24May_2011_Tuesday_tagesschau-4471
内容已经存在，忽略： ./examples/train/25January_2010_Monday_heute-1464
内容已经存在，忽略： ./examples/train/23May_2011_Monday_heute-6456
内容已经存在，忽略： ./examples/train/10December_2011_Saturday_tagesschau-5411
内容已经存在，忽略： ./examples/train/09August_2010_Monday_tagesschau-2995
内容已经存在，忽略： ./examples/train/05December_2011_Monday_heute-7429
内容已经存在，忽略： ./examples/train/05May_2011_Thursday_heute-3751
内容已经存在，忽略： ./examples/train/14June_2010_Monday_tagesschau-3708
内容已经存在，忽略： ./examples/train/20December_2010_Monday_tagesschau-3212
内容已经存在，忽略： ./examples/train/10January_2011_Monday_heute-1372
内容已经存在，忽略： ./examples/train/10August_2010_Tuesday_heute-1450
内容已经存在，忽略： ./examples/train/07December_2009_Monday_tagesschau-7099
内容已经存在，忽略： ./examples/train/24January_2011_Monday_heute-6973
内容已经存在，忽略： ./examples/train/21November_2010_Sunday_tagesschau-5671
内容已经存在，忽略： ./examples/train/06January_2011_Thursday_tagesschau-5651
内容已经存在，忽略： ./examples/train/04April_2010_Sunday_tagesschau-4576
内容已经存在，忽略： ./examples/train/24April_2010_Saturday_tagesschau-3965
内容已经存在，忽略： ./examples/train/30October_2009_Friday_tagesschau-2269
内容已经存在，忽略： ./examples/train/25October_2010_Monday_heute-2048
内容已经存在，忽略： ./examples/train/18April_2010_Sunday_tagesschau-6657
内容已经存在，忽略： ./examples/train/30September_2009_Wednesday_tagesschau-1769
内容已经存在，忽略： ./examples/train/02December_2011_Friday_tagesschau-8014
内容已经存在，忽略： ./examples/train/12April_2010_Monday_heute-7696
内容已经存在，忽略： ./examples/train/12January_2010_Tuesday_heute-2547
内容已经存在，忽略： ./examples/train/28November_2011_Monday_tagesschau-169
内容已经存在，忽略： ./examples/train/05July_2010_Monday_tagesschau-1203
内容已经存在，忽略： ./examples/train/30May_2010_Sunday_tagesschau-4161
内容已经存在，忽略： ./examples/train/26April_2010_Monday_heute-3397
内容已经存在，忽略： ./examples/train/02October_2010_Saturday_tagesschau-1302
内容已经存在，忽略： ./examples/train/13August_2009_Thursday_tagesschau-8486
内容已经存在，忽略： ./examples/train/11November_2009_Wednesday_tagesschau-8038
内容已经存在，忽略： ./examples/train/24June_2011_Friday_tagesschau-4074
内容已经存在，忽略： ./examples/train/20March_2010_Saturday_heute-7280
内容已经存在，忽略： ./examples/train/02October_2012_Tuesday_heute-3502
内容已经存在，忽略： ./examples/train/22February_2010_Monday_tagesschau-3244
内容已经存在，忽略： ./examples/train/20May_2010_Thursday_tagesschau-3157
内容已经存在，忽略： ./examples/train/27June_2011_Monday_heute-2466
内容已经存在，忽略： ./examples/train/12March_2011_Saturday_tagesschau-2453
内容已经存在，忽略： ./examples/train/15December_2010_Wednesday_tagesschau-42
内容已经存在，忽略： ./examples/train/22January_2010_Friday_tagesschau-902
内容已经存在，忽略： ./examples/train/23July_2011_Saturday_tagesschau-5502
内容已经存在，忽略： ./examples/train/07October_2010_Thursday_tagesschau-4123
内容已经存在，忽略： ./examples/train/22September_2010_Wednesday_heute-2619
内容已经存在，忽略： ./examples/train/09January_2010_Saturday_tagesschau-1922
内容已经存在，忽略： ./examples/train/20January_2010_Wednesday_heute-816
内容已经存在，忽略： ./examples/train/03September_2010_Friday_tagesschau-8434
内容已经存在，忽略： ./examples/train/06July_2011_Wednesday_tagesschau-6866
内容已经存在，忽略： ./examples/train/03November_2010_Wednesday_tagesschau-6501
内容已经存在，忽略： ./examples/train/05May_2010_Wednesday_tagesschau-3351
内容已经存在，忽略： ./examples/train/09August_2010_Monday_tagesschau-2997
内容已经存在，忽略： ./examples/train/29September_2010_Wednesday_tagesschau-1785
内容已经存在，忽略： ./examples/train/29September_2010_Wednesday_heute-5459
内容已经存在，忽略： ./examples/train/23February_2010_Tuesday_tagesschau-192
内容已经存在，忽略： ./examples/train/19October_2010_Tuesday_heute-8056
内容已经存在，忽略： ./examples/train/24December_2010_Friday_tagesschau-5132
内容已经存在，忽略： ./examples/train/04May_2011_Wednesday_heute-3992
内容已经存在，忽略： ./examples/train/04March_2011_Friday_tagesschau-2839
内容已经存在，忽略： ./examples/train/30September_2009_Wednesday_tagesschau-1771
内容已经存在，忽略： ./examples/train/27May_2010_Thursday_heute-1226
内容已经存在，忽略： ./examples/train/16June_2010_Wednesday_heute-8314
内容已经存在，忽略： ./examples/train/29January_2013_Tuesday_tagesschau-880
内容已经存在，忽略： ./examples/train/07October_2010_Thursday_heute-8573
内容已经存在，忽略： ./examples/train/24December_2010_Friday_tagesschau-5130
内容已经存在，忽略： ./examples/train/08February_2010_Monday_tagesschau-2803
内容已经存在，忽略： ./examples/train/19July_2009_Sunday_tagesschau-1572
内容已经存在，忽略： ./examples/train/19October_2010_Tuesday_heute-8048
内容已经存在，忽略： ./examples/train/26May_2010_Wednesday_tagesschau-6717
内容已经存在，忽略： ./examples/train/23February_2010_Tuesday_heute-6253
内容已经存在，忽略： ./examples/train/06December_2010_Monday_heute-6211
内容已经存在，忽略： ./examples/train/26October_2010_Tuesday_tagesschau-5791
内容已经存在，忽略： ./examples/train/01August_2011_Monday_heute-4863
内容已经存在，忽略： ./examples/train/17June_2010_Thursday_tagesschau-3905
内容已经存在，忽略： ./examples/train/25August_2009_Tuesday_heute-3301
内容已经存在，忽略： ./examples/train/07December_2011_Wednesday_heute-7803
内容已经存在，忽略： ./examples/train/02December_2010_Thursday_heute-8450
内容已经存在，忽略： ./examples/train/04January_2010_Monday_heute-7405
内容已经存在，忽略： ./examples/train/20September_2010_Monday_tagesschau-7032
内容已经存在，忽略： ./examples/train/13February_2010_Saturday_tagesschau-6045
内容已经存在，忽略： ./examples/train/01February_2011_Tuesday_heute-4640
内容已经存在，忽略： ./examples/train/25March_2011_Friday_tagesschau-5829
内容已经存在，忽略： ./examples/train/10December_2011_Saturday_tagesschau-5419
内容已经存在，忽略： ./examples/train/08April_2010_Thursday_heute-4005
内容已经存在，忽略： ./examples/train/04April_2011_Monday_heute-964
内容已经存在，忽略： ./examples/train/02August_2010_Monday_heute-1244
内容已经存在，忽略： ./examples/train/15April_2010_Thursday_tagesschau-5778
内容已经存在，忽略： ./examples/train/04December_2010_Saturday_tagesschau-4847
内容已经存在，忽略： ./examples/train/17May_2010_Monday_heute-5569
内容已经存在，忽略： ./examples/train/17February_2010_Wednesday_tagesschau-4939
内容已经存在，忽略： ./examples/train/25August_2009_Tuesday_heute-3319
内容已经存在，忽略： ./examples/train/24November_2011_Thursday_heute-225
内容已经存在，忽略： ./examples/train/11May_2011_Wednesday_tagesschau-7018
内容已经存在，忽略： ./examples/train/09August_2010_Monday_heute-5895
内容已经存在，忽略： ./examples/train/17July_2009_Friday_tagesschau-5116
内容已经存在，忽略： ./examples/train/22November_2011_Tuesday_tagesschau-3652
内容已经存在，忽略： ./examples/train/11August_2011_Thursday_heute-3192
内容已经存在，忽略： ./examples/train/29October_2009_Thursday_tagesschau-1015
内容已经存在，忽略： ./examples/train/06December_2011_Tuesday_heute-1544
内容已经存在，忽略： ./examples/train/26August_2009_Wednesday_heute-6557
内容已经存在，忽略： ./examples/train/06April_2010_Tuesday_heute-5983
内容已经存在，忽略： ./examples/train/28October_2010_Thursday_tagesschau-5924
内容已经存在，忽略： ./examples/train/25January_2010_Monday_tagesschau-8304
内容已经存在，忽略： ./examples/train/25August_2009_Tuesday_tagesschau-7087
内容已经存在，忽略： ./examples/train/18January_2011_Tuesday_heute-5470
内容已经存在，忽略： ./examples/train/04May_2010_Tuesday_tagesschau-7843
内容已经存在，忽略： ./examples/train/22April_2010_Thursday_tagesschau-3731
内容已经存在，忽略： ./examples/train/20January_2011_Thursday_tagesschau-3071
内容已经存在，忽略： ./examples/train/05July_2010_Monday_heute-8699
内容已经存在，忽略： ./examples/train/18January_2011_Tuesday_heute-5476
内容已经存在，忽略： ./examples/train/07February_2011_Monday_heute-4667
内容已经存在，忽略： ./examples/train/14July_2010_Wednesday_heute-6194
内容已经存在，忽略： ./examples/train/16July_2011_Saturday_tagesschau-7532
内容已经存在，忽略： ./examples/train/13December_2009_Sunday_tagesschau-6813
内容已经存在，忽略： ./examples/train/21April_2010_Wednesday_tagesschau-6229
内容已经存在，忽略： ./examples/train/25July_2010_Sunday_tagesschau-4780
内容已经存在，忽略： ./examples/train/23May_2010_Sunday_tagesschau-4339
内容已经存在，忽略： ./examples/train/27May_2010_Thursday_heute-1234
内容已经存在，忽略： ./examples/train/25July_2011_Monday_heute-8225
内容已经存在，忽略： ./examples/train/24June_2010_Thursday_tagesschau-7302
内容已经存在，忽略： ./examples/train/21January_2011_Friday_tagesschau-4898
内容已经存在，忽略： ./examples/train/02October_2012_Tuesday_heute-3495
内容已经存在，忽略： ./examples/train/18November_2011_Friday_tagesschau-2151
内容已经存在，忽略： ./examples/train/30July_2010_Friday_tagesschau-1511
内容已经存在，忽略： ./examples/train/02February_2010_Tuesday_tagesschau-7958
内容已经存在，忽略： ./examples/train/17February_2010_Wednesday_tagesschau-4934
内容已经存在，忽略： ./examples/train/09January_2010_Saturday_tagesschau-1930
内容已经存在，忽略： ./examples/train/05October_2010_Tuesday_heute-1222
内容已经存在，忽略： ./examples/train/18January_2010_Monday_tagesschau-4018
内容已经存在，忽略： ./examples/train/18February_2011_Friday_tagesschau-6827
内容已经存在，忽略： ./examples/train/05April_2011_Tuesday_tagesschau-6425
内容已经存在，忽略： ./examples/train/29September_2010_Wednesday_heute-5455
内容已经存在，忽略： ./examples/train/17June_2010_Thursday_tagesschau-3902
内容已经存在，忽略： ./examples/train/25July_2011_Monday_heute-8233
内容已经存在，忽略： ./examples/train/29January_2010_Friday_tagesschau-6492
内容已经存在，忽略： ./examples/train/05March_2011_Saturday_tagesschau-5441
内容已经存在，忽略： ./examples/train/23May_2010_Sunday_tagesschau-4342
内容已经存在，忽略： ./examples/train/14December_2009_Monday_tagesschau-2090
内容已经存在，忽略： ./examples/train/29September_2009_Tuesday_tagesschau-8537
内容已经存在，忽略： ./examples/train/14July_2010_Wednesday_heute-6191
内容已经存在，忽略： ./examples/train/09November_2010_Tuesday_heute-5834
内容已经存在，忽略： ./examples/train/11December_2010_Saturday_tagesschau-8665
内容已经存在，忽略： ./examples/train/11January_2011_Tuesday_heute-8181
内容已经存在，忽略： ./examples/train/04December_2011_Sunday_tagesschau-7786
内容已经存在，忽略： ./examples/train/12November_2009_Thursday_tagesschau-6768
内容已经存在，忽略： ./examples/train/01August_2011_Monday_heute-4859
内容已经存在，忽略： ./examples/train/21March_2011_Monday_tagesschau-4418
内容已经存在，忽略： ./examples/train/25August_2009_Tuesday_heute-3318
内容已经存在，忽略： ./examples/train/19January_2011_Wednesday_heute-2682
内容已经存在，忽略： ./examples/train/30December_2010_Thursday_tagesschau-2230
内容已经存在，忽略： ./examples/train/28December_2011_Wednesday_tagesschau-1914
内容已经存在，忽略： ./examples/train/28October_2010_Thursday_heute-334
内容已经存在，忽略： ./examples/train/08February_2010_Monday_heute-1503
内容已经存在，忽略： ./examples/train/26January_2013_Saturday_tagesschau-7346
内容已经存在，忽略： ./examples/train/23February_2011_Wednesday_heute-3771
内容已经存在，忽略： ./examples/train/14October_2010_Thursday_heute-2190
内容已经存在，忽略： ./examples/train/28September_2010_Tuesday_tagesschau-344
内容已经存在，忽略： ./examples/train/16June_2010_Wednesday_heute-8328
内容已经存在，忽略： ./examples/train/28January_2013_Monday_heute-8006
内容已经存在，忽略： ./examples/train/24May_2011_Tuesday_heute-6071
内容已经存在，忽略： ./examples/train/28August_2009_Friday_tagesschau-5870
内容已经存在，忽略： ./examples/train/31May_2011_Tuesday_tagesschau-4298
内容已经存在，忽略： ./examples/train/06March_2011_Sunday_tagesschau-3896
内容已经存在，忽略： ./examples/train/04April_2011_Monday_heute-972
内容已经存在，忽略： ./examples/train/20October_2010_Wednesday_heute-8495
内容已经存在，忽略： ./examples/train/29August_2009_Saturday_tagesschau-5025
内容已经存在，忽略： ./examples/train/09July_2011_Saturday_tagesschau-3292
内容已经存在，忽略： ./examples/train/04March_2011_Friday_tagesschau-2842
内容已经存在，忽略： ./examples/train/10May_2010_Monday_tagesschau-6855
内容已经存在，忽略： ./examples/train/22November_2011_Tuesday_heute-4649
内容已经存在，忽略： ./examples/train/18January_2010_Monday_heute-3857
内容已经存在，忽略： ./examples/train/05April_2010_Monday_tagesschau-3553
内容已经存在，忽略： ./examples/train/05January_2011_Wednesday_heute-3535
内容已经存在，忽略： ./examples/train/12January_2010_Tuesday_heute-2552
内容已经存在，忽略： ./examples/train/06January_2010_Wednesday_tagesschau-7730
内容已经存在，忽略： ./examples/train/08July_2010_Thursday_tagesschau-5380
内容已经存在，忽略： ./examples/train/16May_2010_Sunday_tagesschau-5228
内容已经存在，忽略： ./examples/train/08November_2010_Monday_heute-6884
内容已经存在，忽略： ./examples/train/18February_2010_Thursday_tagesschau-4525
内容已经存在，忽略： ./examples/train/19February_2010_Friday_tagesschau-4100
内容已经存在，忽略： ./examples/train/26April_2010_Monday_heute-3395
内容已经存在，忽略： ./examples/train/15July_2011_Friday_tagesschau-8721
内容已经存在，忽略： ./examples/train/26January_2013_Saturday_tagesschau-7354
内容已经存在，忽略： ./examples/train/29September_2011_Thursday_heute-4232
内容已经存在，忽略： ./examples/train/18May_2010_Tuesday_tagesschau-3580
内容已经存在，忽略： ./examples/train/09February_2011_Wednesday_tagesschau-2510
内容已经存在，忽略： ./examples/train/27April_2010_Tuesday_heute-1031
内容已经存在，忽略： ./examples/train/28May_2010_Friday_tagesschau-7500
内容已经存在，忽略： ./examples/train/30March_2011_Wednesday_tagesschau-7128
内容已经存在，忽略： ./examples/train/03November_2010_Wednesday_tagesschau-6500
内容已经存在，忽略： ./examples/train/27February_2011_Sunday_tagesschau-6115
内容已经存在，忽略： ./examples/train/23August_2010_Monday_tagesschau-4433
内容已经存在，忽略： ./examples/train/22February_2010_Monday_tagesschau-3251
内容已经存在，忽略： ./examples/train/09July_2010_Friday_tagesschau-599
内容已经存在，忽略： ./examples/train/06October_2010_Wednesday_tagesschau-635
内容已经存在，忽略： ./examples/train/20January_2010_Wednesday_heute-805
内容已经存在，忽略： ./examples/train/19January_2011_Wednesday_tagesschau-7079
内容已经存在，忽略： ./examples/train/05April_2011_Tuesday_heute-5121
内容已经存在，忽略： ./examples/train/02October_2012_Tuesday_heute-3500
内容已经存在，忽略： ./examples/train/14August_2011_Sunday_tagesschau-3236
内容已经存在，忽略： ./examples/train/11September_2009_Friday_tagesschau-8775
内容已经存在，忽略： ./examples/train/23August_2010_Monday_heute-5324
内容已经存在，忽略： ./examples/train/22February_2010_Monday_tagesschau-3252
内容已经存在，忽略： ./examples/train/25October_2010_Monday_heute-2050
内容已经存在，忽略： ./examples/train/28September_2010_Tuesday_heute-8354
内容已经存在，忽略： ./examples/train/03March_2011_Thursday_tagesschau-7057
内容已经存在，忽略： ./examples/train/24February_2011_Thursday_heute-6679
内容已经存在，忽略： ./examples/train/27April_2010_Tuesday_tagesschau-4511
内容已经存在，忽略： ./examples/train/24August_2010_Tuesday_heute-3014
内容已经存在，忽略： ./examples/train/07June_2010_Monday_heute-2325
内容已经存在，忽略： ./examples/train/01June_2011_Wednesday_heute-953
内容已经存在，忽略： ./examples/train/25January_2013_Friday_tagesschau-7812
内容已经存在，忽略： ./examples/train/04January_2010_Monday_heute-7403
内容已经存在，忽略： ./examples/train/29September_2011_Thursday_tagesschau-7228
内容已经存在，忽略： ./examples/train/02June_2010_Wednesday_heute-6785
内容已经存在，忽略： ./examples/train/27May_2011_Friday_tagesschau-8123
内容已经存在，忽略： ./examples/train/24June_2011_Friday_tagesschau-4067
内容已经存在，忽略： ./examples/train/23May_2011_Monday_tagesschau-1105
内容已经存在，忽略： ./examples/train/19March_2011_Saturday_tagesschau-4687
内容已经存在，忽略： ./examples/train/23February_2011_Wednesday_heute-3776
内容已经存在，忽略： ./examples/train/11September_2009_Friday_tagesschau-8774
内容已经存在，忽略： ./examples/train/18January_2010_Monday_heute-3861
内容已经存在，忽略： ./examples/train/02September_2010_Thursday_heute-3451
内容已经存在，忽略： ./examples/train/12July_2009_Sunday_tagesschau-2572
内容已经存在，忽略： ./examples/train/12March_2011_Saturday_tagesschau-2456
内容已经存在，忽略： ./examples/train/12July_2010_Monday_tagesschau-375
内容已经存在，忽略： ./examples/train/05October_2010_Tuesday_heute-1217
内容已经存在，忽略： ./examples/train/23December_2010_Thursday_tagesschau-1436
内容已经存在，忽略： ./examples/train/16July_2011_Saturday_tagesschau-7530
内容已经存在，忽略： ./examples/train/13December_2009_Sunday_tagesschau-6812
内容已经存在，忽略： ./examples/train/31January_2013_Thursday_tagesschau-2711
内容已经存在，忽略： ./examples/train/11April_2010_Sunday_tagesschau-8640
内容已经存在，忽略： ./examples/train/29August_2009_Saturday_tagesschau-5029
内容已经存在，忽略： ./examples/train/11August_2009_Tuesday_tagesschau-4349
内容已经存在，忽略： ./examples/train/03March_2011_Thursday_tagesschau-7061
内容已经存在，忽略： ./examples/train/14October_2010_Thursday_heute-2188
内容已经存在，忽略： ./examples/train/06April_2010_Tuesday_tagesschau-307
内容已经存在，忽略： ./examples/train/30March_2010_Tuesday_heute-7568
内容已经存在，忽略： ./examples/train/22June_2011_Wednesday_heute-4380
内容已经存在，忽略： ./examples/train/26June_2010_Saturday_tagesschau-719
内容已经存在，忽略： ./examples/train/04April_2011_Monday_heute-967
内容已经存在，忽略： ./examples/train/13September_2010_Monday_heute-7183
内容已经存在，忽略： ./examples/train/04July_2011_Monday_heute-6450
内容已经存在，忽略： ./examples/train/31August_2010_Tuesday_tagesschau-5170
内容已经存在，忽略： ./examples/train/22April_2010_Thursday_tagesschau-3729
内容已经存在，忽略： ./examples/train/01November_2010_Monday_heute-1717
内容已经存在，忽略： ./examples/train/08November_2010_Monday_heute-6887
内容已经存在，忽略： ./examples/train/20November_2011_Sunday_tagesschau-545
内容已经存在，忽略： ./examples/train/01July_2010_Thursday_tagesschau-8794
内容已经存在，忽略： ./examples/train/09February_2011_Wednesday_heute-7545
内容已经存在，忽略： ./examples/train/09February_2010_Tuesday_tagesschau-4502
内容已经存在，忽略： ./examples/train/14February_2010_Sunday_tagesschau-3842
内容已经存在，忽略： ./examples/train/22November_2011_Tuesday_tagesschau-3651
内容已经存在，忽略： ./examples/train/31May_2011_Tuesday_heute-1895
内容已经存在，忽略： ./examples/train/06April_2010_Tuesday_tagesschau-312
内容已经存在，忽略： ./examples/train/09February_2011_Wednesday_heute-7546
内容已经存在，忽略： ./examples/train/17April_2011_Sunday_tagesschau-2157
内容已经存在，忽略： ./examples/train/07September_2010_Tuesday_tagesschau-1957
内容已经存在，忽略： ./examples/train/31May_2011_Tuesday_heute-1897
内容已经存在，忽略： ./examples/train/30August_2011_Tuesday_heute-785
内容已经存在，忽略： ./examples/train/29July_2010_Thursday_heute-7740
内容已经存在，忽略： ./examples/train/14December_2009_Monday_tagesschau-2089
内容已经存在，忽略： ./examples/train/23July_2010_Friday_tagesschau-583
内容已经存在，忽略： ./examples/train/06May_2010_Thursday_tagesschau-5745
内容已经存在，忽略： ./examples/train/01April_2011_Friday_tagesschau-3380
内容已经存在，忽略： ./examples/train/21November_2011_Monday_heute-5435
内容已经存在，忽略： ./examples/train/01August_2011_Monday_heute-4851
内容已经存在，忽略： ./examples/train/24February_2011_Thursday_tagesschau-4762
内容已经存在，忽略： ./examples/train/02March_2011_Wednesday_tagesschau-4707
内容已经存在，忽略： ./examples/train/26August_2010_Thursday_tagesschau-4271
内容已经存在，忽略： ./examples/train/02July_2009_Thursday_tagesschau-2864
内容已经存在，忽略： ./examples/train/27June_2011_Monday_heute-2471
内容已经存在，忽略： ./examples/train/24June_2010_Thursday_tagesschau-7306
内容已经存在，忽略： ./examples/train/30November_2011_Wednesday_tagesschau-6287
内容已经存在，忽略： ./examples/train/28February_2011_Monday_tagesschau-4979
内容已经存在，忽略： ./examples/train/10April_2010_Saturday_tagesschau-4133
内容已经存在，忽略： ./examples/train/28March_2011_Monday_tagesschau-2290
内容已经存在，忽略： ./examples/train/06August_2010_Friday_tagesschau-2175
内容已经存在，忽略： ./examples/train/22January_2011_Saturday_tagesschau-2076
内容已经存在，忽略： ./examples/train/20March_2010_Saturday_heute-7283
内容已经存在，忽略： ./examples/train/23February_2010_Tuesday_heute-6247
内容已经存在，忽略： ./examples/train/04October_2010_Monday_heute-5948
内容已经存在，忽略： ./examples/train/25August_2009_Tuesday_heute-3310
内容已经存在，忽略： ./examples/train/04March_2011_Friday_tagesschau-2837
内容已经存在，忽略： ./examples/train/02November_2010_Tuesday_heute-2756
内容已经存在，忽略： ./examples/train/03February_2010_Wednesday_heute-2351
内容已经存在，忽略： ./examples/train/02March_2011_Wednesday_tagesschau-4703
内容已经存在，忽略： ./examples/train/25October_2010_Monday_heute-2045
内容已经存在，忽略： ./examples/train/06December_2011_Tuesday_heute-1548
内容已经存在，忽略： ./examples/train/06April_2010_Tuesday_tagesschau-304
内容已经存在，忽略： ./examples/train/29March_2010_Monday_tagesschau-8390
内容已经存在，忽略： ./examples/train/11January_2011_Tuesday_tagesschau-8031
内容已经存在，忽略： ./examples/train/07December_2010_Tuesday_heute-5638
内容已经存在，忽略： ./examples/train/25July_2010_Sunday_tagesschau-4775
内容已经存在，忽略： ./examples/train/01July_2009_Wednesday_tagesschau-4553
内容已经存在，忽略： ./examples/train/21October_2010_Thursday_tagesschau-1818
内容已经存在，忽略： ./examples/train/28October_2010_Thursday_heute-326
内容已经存在，忽略： ./examples/train/31August_2010_Tuesday_tagesschau-5173
内容已经存在，忽略： ./examples/train/07February_2011_Monday_heute-4662
内容已经存在，忽略： ./examples/train/25January_2011_Tuesday_tagesschau-2985
内容已经存在，忽略： ./examples/train/09December_2009_Wednesday_tagesschau-2560
内容已经存在，忽略： ./examples/train/19May_2010_Wednesday_heute-1116
内容已经存在，忽略： ./examples/train/01March_2011_Tuesday_tagesschau-2208
内容已经存在，忽略： ./examples/train/28September_2010_Tuesday_tagesschau-348
内容已经存在，忽略： ./examples/train/27August_2009_Thursday_tagesschau-3272
内容已经存在，忽略： ./examples/train/29May_2011_Sunday_tagesschau-756
内容已经存在，忽略： ./examples/train/25November_2009_Wednesday_tagesschau-7679
内容已经存在，忽略： ./examples/train/28August_2009_Friday_tagesschau-5877
内容已经存在，忽略： ./examples/train/25July_2010_Sunday_tagesschau-4778
内容已经存在，忽略： ./examples/train/11April_2010_Sunday_tagesschau-8642
内容已经存在，忽略： ./examples/train/28August_2010_Saturday_tagesschau-8184
内容已经存在，忽略： ./examples/train/06April_2010_Tuesday_heute-5973
内容已经存在，忽略： ./examples/train/29April_2010_Thursday_tagesschau-4876
内容已经存在，忽略： ./examples/train/01August_2011_Monday_heute-4852
内容已经存在，忽略： ./examples/train/25May_2010_Tuesday_heute-3759
内容已经存在，忽略： ./examples/train/24August_2010_Tuesday_heute-3022
内容已经存在，忽略： ./examples/train/30July_2011_Saturday_tagesschau-8592
内容已经存在，忽略： ./examples/train/07December_2011_Wednesday_heute-7810
内容已经存在，忽略： ./examples/train/15September_2010_Wednesday_heute-2486
内容已经存在，忽略： ./examples/train/08June_2010_Tuesday_heute-1989
内容已经存在，忽略： ./examples/train/01April_2010_Thursday_heute-6696
内容已经存在，忽略： ./examples/train/30November_2009_Monday_tagesschau-5759
内容已经存在，忽略： ./examples/train/11August_2011_Thursday_heute-3196
内容已经存在，忽略： ./examples/train/08June_2010_Tuesday_heute-1995
内容已经存在，忽略： ./examples/train/09November_2010_Tuesday_tagesschau-748
内容已经存在，忽略： ./examples/train/11February_2010_Thursday_tagesschau-8762
内容已经存在，忽略： ./examples/train/09April_2010_Friday_tagesschau-7631
内容已经存在，忽略： ./examples/train/15February_2011_Tuesday_tagesschau-7472
内容已经存在，忽略： ./examples/train/29September_2011_Thursday_heute-4227
内容已经存在，忽略： ./examples/train/24October_2010_Sunday_tagesschau-2900
内容已经存在，忽略： ./examples/train/19January_2011_Wednesday_heute-2685
内容已经存在，忽略： ./examples/train/20July_2010_Tuesday_tagesschau-7879
内容已经存在，忽略： ./examples/train/13January_2010_Wednesday_tagesschau-3830
内容已经存在，忽略： ./examples/train/15June_2010_Tuesday_tagesschau-89
内容已经存在，忽略： ./examples/train/05July_2010_Monday_heute-8694
内容已经存在，忽略： ./examples/train/05January_2010_Tuesday_tagesschau-2670
内容已经存在，忽略： ./examples/train/11August_2010_Wednesday_tagesschau-7
内容已经存在，忽略： ./examples/train/10January_2011_Monday_heute-1374
内容已经存在，忽略： ./examples/train/25July_2010_Sunday_tagesschau-4773
内容已经存在，忽略： ./examples/train/09February_2010_Tuesday_tagesschau-4494
内容已经存在，忽略： ./examples/train/29April_2010_Thursday_heute-8628
内容已经存在，忽略： ./examples/train/02December_2009_Wednesday_tagesschau-4043
内容已经存在，忽略： ./examples/train/23February_2011_Wednesday_tagesschau-152
内容已经存在，忽略： ./examples/train/29July_2010_Thursday_tagesschau-7415
内容已经存在，忽略： ./examples/train/17February_2011_Thursday_tagesschau-6312"""

    # 将文本按行分割，并计算行数
    line_count = len(text.splitlines())
    print("行数：", line_count)
