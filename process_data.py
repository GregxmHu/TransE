entity=[]
relation=[]
idtriplets=[]
with open("triplets.tsv","r") as f:
    for item in f:
        h,r,t=item.strip('\n').split(',')
        if h not in entity:
            entity.append(h)
        if t not in entity:
            entity.append(t)
        if r not in entity:
            relation.append(r)
        idtriplets.append(
            (entity.index(h),relation.index(r),entity.index(t))
            )

with open("entity2id.tsv","w") as f:
    for i in range(len(entity)):
        f.write(str(i)+','+entity[i]+'\n')

with open("relation2id.tsv","w") as f:
    for i in range(len(relation)):
        f.write(str(i)+','+relation[i]+'\n')

with open("id_triplets.tsv","w") as f:
    for item in idtriplets:
        f.write(str(item[0])+','+str(item[1])+','+str(item[2])+'\n')


