import network

train_loss,cost,optimizer,merged,global_step,train_label,train_out=network.generate_network(train=False)

test_accuracy_list=[]
test_loss_list=[]
for ll in range(15000//BATCH_SIZE):
	test_output_,test_label_,test_loss_=sess.run([test_output,test_label,test_loss],feed_dict={keep_prob:1})
	accuracy=compute_accuracy(output=test_output_,label=test_label_)
	test_accuracy_list.append(accuracy)
	test_loss_list.append(test_loss_)
print '%dEPOCHS:test_loss is %.6f, testing accuracy is %.6f\n'%(ii,np.mean(test_loss_list),np.mean(test_accuracy_list))

print '%dEPOCHS: testing accuracy is %.6f'%(ii,np.mean(test_accuracy_list))


