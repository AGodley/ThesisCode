
# Creates the dictionary
analysis_x = {}

# Initializes the first two patterns
analysis_x['1'] = 0
analysis_x['11'] = 0

# Generates all patterns up to order n+2
n = 10
for i in range(2**1, 2**(n+1)):
    analysis_x['1'+bin(i)[3:]+'1'] = 0
print(analysis_x)