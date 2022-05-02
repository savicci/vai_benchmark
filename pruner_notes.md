flow:

- ratio must increase every iteration
- sometimes increasing ratio will not make model sparser, gotta break here

loop:
- make x iterations max
- in every iteration:
- check if accuracy is less than desired value (must be less than original model accuracy)
- if it is, then save previous model as final
- if not, start pruning with some ratio
- increase ratio every loop still must be less than 1, find some function that converges to 1 first fast, then slowly
- save checkpoints
- 
