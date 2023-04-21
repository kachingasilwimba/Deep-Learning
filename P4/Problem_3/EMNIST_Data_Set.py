import EMNIST_Loader2
tr_d, va_d, te_d = EMNIST_Loader2.load_data()
import matplotlib.pyplot as plt
print("type(tr_d):",type(tr_d))
print()
print("len(tr_d):", len(tr_d))
print()
print("tr_d[0]:", tr_d[0])
print()
print("len(tr_d[0]):", len(tr_d[0]))
print()
print("tr_d[0].shape:",tr_d[0].shape)
print()
print("type(tr_d[0]):",type(tr_d[0]))
print()
print("tr_d[0][0].shape:",tr_d[0][0].shape)
print("")
print("print training labels tr_d[1]:", tr_d[1])
print("")
j = 99896
print("Plot training image j =", j)
print("image = tr_d[0][j]")
image = tr_d[0][j]
print('image = tr_d[0][j].shape = ',tr_d[0][j].shape )
print()
print("Reshape the (784L,) array into an (28,28) array")
print()
imagenew = image.reshape(28,28)
print("Print jth image from the dataset:")
plt.imshow(imagenew)
plt.show()
label_num = tr_d[1][j]
print("label_num:", label_num)         

result_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
               10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',
               19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',
               28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',
               36:'a',37:'b',38:'d',39:'e',40:'f',41:'g',42:'h',43:'n',44:'q',
               45:'r',46:'t'
               }

print('label is',result_dict[label_num])