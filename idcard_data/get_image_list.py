import os.path as osp

def short_name(name):
    spl = name.split('/')

    return spl[-2]+'/'+spl[-1]

fn_list = './identities-dataset_list.txt'

fn_img_list = './img_list_all_with_id.txt'
fn_card_list = './img_list_idcard_with_id.txt'
fn_scene_list = './img_list_scene_with_id.txt'

fn_name_list = './name_list_with_id.txt'

img_list = []
#card_img_list = []
#scene_img_list = []

last_name = None
name_list = []
#name_list2 = []

name_id_cnt = -1

fp = open(fn_list, 'r')
fp_img_list = open(fn_img_list, 'w')
fp_card_list = open(fn_card_list, 'w')
fp_scene_list = open(fn_scene_list, 'w')

for line in fp:
    line = line.strip()
    spl = line.split('/')

    img_name = spl[-2] + '/' + spl[-1]
    img_list.append(line)

    name = spl[-2]
#    if name not in name_list2:
#        name_list2.append(name)
    if name != last_name:
        name_list.append(name)
        last_name = name
        name_id_cnt += 1

    fp_img_list.write('%s %d\n' % (img_name, name_id_cnt))

    if 'card' in spl[-1]:
#        card_img_list.append(line)
        fp_card_list.write('%s %d\n' % (img_name, name_id_cnt))
    else:
#        scene_img_list.append(line)
        fp_scene_list.write('%s %d\n' % (img_name, name_id_cnt))

fp.close()
fp_img_list.close()
fp_card_list.close()
fp_scene_list.close()

#print "found %d names" % len(name_list2)
print "found %d names" % len(name_list)
#print " found %d names with a card photo" % len(name_list)

fp_name_list = open(fn_name_list, 'w')

for i, name in enumerate(name_list):
    fp_name_list.write('%d %s' % (i, name))
fp_name_list.close()
