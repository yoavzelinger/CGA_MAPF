# import time
#
# start = time.time()
# counter: int = 0
# max_iter = 1_000_000
# # max_iter = 10_000_000
# # max_iter = 100_000_000
# max_iter = 1_000_000_000
# while counter < max_iter:
#   counter+=1
# print(counter)
# print(f'{time.time() - start : .2f}')


import time

start = time.time()
max_iter = 1_000_000_000
counter: int = 0
while counter < max_iter:
  counter+=1
print(counter)
print(f'{time.time() - start : .2f}')