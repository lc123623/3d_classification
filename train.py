import network

train_loss,cost,optimizer,merged,global_step,train_label,train_out=network.generate_network(train=true)

for ii in range(EPOCHS):
	logfile=open('log.txt','a')
	train_accuracy_list=[]
	for jj in range(60000//BATCH_SIZE):
		train_loss_,cost_,_,merged_,global_step_,train_label_,train_out_=sess.run([train_loss,cost,optimizer,merged,global_step,train_label,train_out],feed_dict={keep_prob:0.5})
		print '%dEPOCHS %diteration: training loss is %.6f'%(ii,jj,train_loss_)
		train_accuracy=compute_accuracy(train_out_,train_label_)
		train_accuracy_list.append(train_accuracy)

		if global_step_>100:
			writer.add_summary(merged_,global_step_)
	logfile.write('%dEPOCHS: training loss is %.6f,  total_loss:%.6f,train_accuracy:%.6f\n'%(ii,train_loss_,cost_,np.mean(train_accuracy_list)))
	logfile.close()
	if ii%20==0:
		saver_path=saver.save(sess,'./model/model.ckpt',global_step=global_step_)

