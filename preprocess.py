def read_lines(filename):
	return open(filename).read().split('\n')[:-1]

def create_data_set(lines,context_size):
	contexts = []
	responses = []
	sent = ''
	count = 0
	for i in range(context_size,len(lines),context_size+1):	
		for j in range(context_size,0,-1):
			sent += ' ' + lines[i-j]	
		contexts.append(sent)
		responses.append(lines[i])
		sent = ''	
		count += 1
	return (contexts,responses)

def preprocess_data(filename,context_szie):	
    	lines = read_lines(filename)	
	return create_data_set(lines,context_size)
	
def save_data(contexts,responses,train_set_path,train_set_target_path,test_set_path,test_set_target_path,test_set_percent):
	test_set_size = int(len(contexts)*test_set_percent)
	train_set_size = len(contexts) - test_set_size
	training_set = contexts[:train_set_size]
	test_set = contexts[train_set_size:len(contexts)]
	training_set_target = responses[:len(training_set)]
	test_set_target = responses[train_set_size:len(responses)]
	save_in_file(training_set,train_set_path)
	save_in_file(training_set_target,train_set_target_path)
	save_in_file(test_set,test_set_path)
	save_in_file(test_set_target,test_set_target_path)

def save_in_file(data_set,filename):
	with open(filename,'w') as data_file:
		for line in data_set:
			data_file.write(line)
			data_file.write('\n')

if __name__ == "__main__":
	filename='data/chat.txt'
	context_size = 1
	train_set_path = 'twitter_train_set'
	train_set_target_path = 'twitter_train_set_target'
	test_set_path = 'twitter_test_set'
	test_set_target_path = 'twitter_test_set_target'
	test_set_percent = 0.2
	(contexts,responses) = preprocess_data(filename,context_size)
	save_data(contexts,responses,train_set_path,train_set_target_path,test_set_path,test_set_target_path,test_set_percent)

