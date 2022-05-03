ptq without finetuning is the worst but sometimes its like 1,2 % of accuracy loss
we can add finetuning that trains network again on training set, adds some accuracy
if accuracy after quantizing is bad(mobilenet for example) we can use qat which further increases accuracy

for qat it is good idea to start with pretrained model and add some epochs to fine tune - fmnist example

TODO: test qat on not trained example to check how much really is lost. On ptq -> qat we had increase
of accuracy because we doubled epochs to train