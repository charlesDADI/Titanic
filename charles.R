	
	##############################################################
	##			by Charles-Abner Dadi 				##
	##############################################################


set.seed(1234)
library(xts)
library(tseries)
library(randomForest)
library(e1071)
library(nnet)
library(class)
library(ada)
library(caret)
library(gbm)
library(earth)
library(MASS)
library(nFactors)
library(kernlab)

############
##Function##
############
score<-function(Preds,True){
	tab=table(pred=Preds,true=True)
	accurancy=(tab[1,1]+tab[2,2])/sum(tab)
	return(list("confusion"=tab,"accurancy"=accurancy))
}


Cleaning<-function(Data)
{
if('Survived' %in% names(Data))Data[,'Survived']<-as.factor(Data[,'Survived'])
Data[,'Sex']<-as.factor(Data[,'Sex'])
Data[,'Pclass']<-as.factor(Data[,'Pclass'])
Data[,'Age']<-as.numeric(Data[,'Age'])
if(length((which(Data[,'Embarked']=="")>0)))Data[(which(Data[,'Embarked']=="")),'Embarked']="S"		##a coriger
Data[,'Embarked']<-as.factor(Data[,'Embarked'])
Data[,'Fare']<-as.numeric(Data[,'Fare'])
if(length(which(is.na(Data[,'Fare'])))>0){
	reg1<-lm(Fare~Pclass+Age+Embarked+SibSp,data=Data[which(is.na(Data[,'Fare'])==FALSE),])
	Data[which(is.na(Data[,'Fare'])==TRUE),'Fare']=predict(reg1,Data[which(is.na(Data[,'Fare'])==TRUE),])
}
Data<-Data[,!(names(Data) %in% c('Cabin','Ticket'))] #Je retire les ID 
return(Data)
}

TraitementAge<-function(Data)
{

#pour les  Mr( homme/femme on tire plutot > 50 ans)
MrNoAge	<-as.numeric(intersect(rownames(Data)[grep("Mr.",Data[,'Name'])], rownames(Data)[is.na(Data[,'Age'])]))
MrWithAge	<-as.numeric(intersect(rownames(Data)[grep("Mr.",Data[,'Name'])], rownames(Data)[is.na(Data[,'Age'])==FALSE]))
if(length(MrWithAge)>0){
reg_Age	<-lm(Age~Pclass+SibSp+Embarked, data=as.data.frame(Data[MrWithAge,]) )
Data[MrNoAge,'Age'] <- predict(reg_Age,newdata=Data[MrNoAge,])
}


#pour les  Mrs entre 35 et 55 ans
MrsNoAge	<-as.numeric(intersect(rownames(Data)[grep("Mrs.",Data[,'Name'])], rownames(Data)[is.na(Data[,'Age'])]))
MrsWithAge	<-as.numeric(intersect(rownames(Data)[grep("Mrs.",Data[,'Name'])], rownames(Data)[is.na(Data[,'Age'])==FALSE]))
if(length(MrsNoAge)>0){
	reg_Age	<-lm(Age~Pclass+SibSp+Embarked, data=as.data.frame(Data[MrsWithAge,]) )
	Data[MrsNoAge,'Age']	<-	predict(reg_Age,newdata=Data[MrsNoAge,])
}

#pour les  Miss entre 20 et 40 ans
MissNoAge	<-as.numeric(intersect(rownames(Data)[grep("Miss.",Data[,'Name'])], rownames(Data)[is.na(Data[,'Age'])]))
MissWithAge	<-as.numeric(intersect(rownames(Data)[grep("Miss.",Data[,'Name'])], rownames(Data)[is.na(Data[,'Age'])==FALSE]))
if(length(MissNoAge)>0){
	reg_Age	<-lm(Age~Pclass+SibSp+Embarked, data=as.data.frame(Data[MissWithAge,]))
	Data[MissNoAge,'Age']<-predict(reg_Age,newdata=Data[MissNoAge,])
}

reg_Age	<-lm(Age~Sex+Pclass+SibSp+Embarked, data=na.omit(Data))
Data[is.na(Data[,'Age']),'Age']<-predict(reg_Age,Data[which(is.na(Data[,'Age'])==TRUE),])
Data<-Data[,!(names(Data) %in% c('PassengerId','Name'))]

for(k in 1:ncol(Data))if(is.factor(Data[,k])==FALSE) (Data[,k]<-as.numeric(Data[,k]))

return(Data)
}




SelectData<-function(Data, Profile)
{
classe=Profile$classe
sex		<-	Profile$sex
age		<-	Profile$age
siblings	<-	Profile$siblings 
parents	<- 	Profile$parents  
embarked	<-	Profile$embarked
view		<-	Profile$view
indClasse	<-	which(Data[,'Pclass'] %in% classe)
indAge	<-	intersect(which(Data[,'Age']>=age[1]),(which(Data[,'Age']<=age[2])))
indSex	<-	which(Data[,'Sex']%in% sex)
indSiblings	<-	intersect(which(Data[,'SibSp']>=siblings[1]),(which(Data[,'SibSp']<=siblings[2])))
indParents	<-	intersect(which(Data[,'Parch']>=parents[1]),(which(Data[,'Parch']<=parents[2])))
indEmbarked	<-	which(Data[,'Embarked'] %in% embarked)
ind		<-	Reduce(intersect, list(indClasse,indAge ,indSex,indSiblings,indParents))
if(view==TRUE)View(Data[ind,])
return(Data[ind,])
}

LaunchClassification<-function(X,Methods=c('randomForest','ksvm','logistic','AdaBoost','mlp','svm','naiveBayes'))
{
#on cree une liste pour recuperer les predictions des diff�rentes methodes
Preds <- list()
#on cree une liste pour recuperer les accurancy des differentes methods
Acc<-list()
#on cree une liste pour recuperer les modeles d'apprentissage des differentes methods
Models<-list()

#division des donnes pour faire mon propre test
table		<-	c(1:nrow(X))
ind		<-	sample(table,trunc(0.75*nrow(X)),replace=FALSE)
compl		<-	table[-ind]
X_train	<-	X[ind, ]
X_test	<-	X[compl, ]

#formule de classification
Tform <- as.formula('Survived ~ .')	

for (m in Methods){

if(m=="randomForest"){
			K<-5000
			rf   <-randomForest(Tform, data=X_train,ntree=K,importance=TRUE)
			preds <-predict(rf,X_test[,!(names(X_test) %in% c('Survived'))])
			sc	<-score(Preds=preds,True=X_test[,'Survived'])
			plot(varImpPlot(rf, main="Var Importance",col="dark blue"))
			jpeg(filename="Tuning SVM")
			dev.off()
			Preds[[m]]=preds
			Acc[[m]]=sc$accurancy
			Models[[m]]=rf
			print(paste("Accurancy: ",m, " = ",sc$accurancy))
 			}

if(m=='logistic'){
			logi  <-glm(Tform,family = binomial(logit), data = X_train)
			preds <-predict(logi, X_test[,!(names(X_test) %in% c('Survived'))], type = "response")
			preds[preds > 0.5]=1;preds[preds <= 0.5]=0;
			sc	<-score(Preds=preds,True=X_test[,'Survived'])
			Preds[[m]]=preds
			Acc[[m]]=sc$accurancy
			Models[[m]]=logi
			print(paste("Accurancy: ",m, " = ",sc$accurancy))
			}

if(m=='mlp'){
			K<-10
			ann 	<- nnet(Tform,data=X_train, size=K,maxit=100)
			preds	<-predict(ann, X_test[,!(names(X_test) %in% c('Survived'))], type="class")
			sc	<-score(Preds=preds,True=X_test[,'Survived'])
			Preds[[m]]=preds
			Acc[[m]]=sc$accurancy
			Models[[m]]=ann
			print(paste("Accurancy: ",m, " = ",sc$accurancy))
			}

if(m=='AdaBoost'){
			k<-10000
			ad	<-ada(Tform,data=X_train,loss="linear",iter=K)
			preds	<-predict(ad, X_test[,!(names(X_test) %in% c('Survived'))])
			sc	<-score(Preds=preds,True=X_test[,'Survived'])
			Preds[[m]]=preds
			Acc[[m]]=sc$accurancy
			Models[[m]]=ad
			print(paste("Accurancy: ",m, " = ",sc$accurancy))

			}
if(m=='svm'){
#http://www.inside-r.org/packages/cran/e1071/docs/tune	
			obj <- tune(svm, Tform, data = X_train, 
             	 ranges = list(gamma = 10^(-1:1), cost = 2^(0:5)),
              	tunecontrol = tune.control(sampling = "fix")
            	 )
  			summary(obj)
 		 	plot(obj)
		jpeg(filename="Tuning SVM")
		dev.off()
			sv	<-svm(Tform, data=X_train, kernel="linear",type='C-classification', gamma=obj$best.parameters$gamma, cost=obj$ best.parameters$cost,cross=10)
			preds	<-predict(sv,X_test[,!(names(X_test) %in% c('Survived'))])
			sc	<-score(Preds=preds,True=X_test[,'Survived'])
			Preds[[m]]=preds
			Acc[[m]]=sc$accurancy
			Models[[m]]=sv
			print(paste("Accurancy: ",m, " = ",sc$accurancy))

			}

if(m=='ksvm'){
#http://www.inside-r.org/packages/cran/e1071/docs/tune	
			sv	<-ksvm(Tform, data=X_train,type="C-svc" )
			preds	<-predict(sv,X_test[,!(names(X_test) %in% c('Survived'))])
			sc	<-score(Preds=preds,True=X_test[,'Survived'])
			Preds[[m]]=preds
			Acc[[m]]=sc$accurancy
			Models[[m]]=sv
			print(paste("Accurancy: ",m, " = ",sc$accurancy))

			}

if(m=='naiveBayes'){
			nvB 	<- naiveBayes(Tform, data = X_train)
			preds	<-predict(nvB , X_test[,!(names(X_test) %in% c('Survived'))])
			sc	<-score(Preds=preds,True=X_test[,'Survived'])
			Preds[[m]]=preds
			Acc[[m]]=sc$accurancy
			Models[[m]]=nvB
			print(paste("Accurancy: ",m, " = ",sc$accurancy))
		}

if(m=='decisionTrees'){
  			obj <- tune.rpart(Tform, data =  X_train, minsplit = c(5,10,15,20,40))
			summary(obj)
 			plot(obj)
			jpeg(filename="Tuning SVM")
			dev.off()
			dt	<- rpart(Tform, data = X_train)
			preds	<- predict(dt , X_test[,!(names(X_test) %in% c('Survived'))])
			preds	<- ifelse(preds[,1]>0.5,0,1)
			sc	<-score(Preds=preds,True=X_test[,'Survived'])
			Preds[[m]]=preds
			Acc[[m]]=sc$accurancy
			Models[[m]]=dt 
			print(paste("Accurancy: ",m, " = ",sc$accurancy))
			}
	}

return(list(Preds,Acc,Models))
}



PredFromProfile<-function( Train, Test, Profile)
{
TrainSelection<-SelectData(Train,Profile)
TestSelection<-SelectData(Test,Profile)

Training<-LaunchClassification(TrainSelection,Methods=Profile$Methods)

best_method		<- names(Training[[1]])[which(data.frame(Training[[2]])==max(data.frame(Training[[2]])))[1]]
best_accurancy	<- Training[[2]][which(data.frame(Training[[2]])==max(data.frame(Training[[2]])))[1]]
best_model		<- Training[[3]][which(data.frame(Training[[2]])==max(data.frame(Training[[2]])))[1]]

preds<-predict(best_model,TestSelection)
preds_matrix[as.numeric(rownames(TestSelection)),1]<-as.numeric(rownames(TestSelection))
preds_matrix[as.numeric(rownames(TestSelection)),2]<-preds
preds_matrix<<-preds_matrix
#preds_matrix[,2]=as.factor(preds_matrix[,2],labels=c(0,1,3))
}


#######################

###############################
##########Load Data############
###############################

output.path	<<-as.character("C:/Users/charlesabner/Dropbox/Ivan-Charles-Dan/titanic/")
input.path	<<-as.character("C:/Users/charlesabner/Dropbox/Ivan-Charles-Dan/titanic/")

Data			<-	read.csv2(paste(input.path,"train.csv",sep=""),header=TRUE,sep=",",stringsAsFactors=FALSE,skip=0)
Data_submit	<-	read.csv2(paste(input.path,"test.csv",sep=""),header=TRUE,sep=",",stringsAsFactors=FALSE,skip=0)
N_data		<-	nrow(Data)


Data	<-	Cleaning(Data)
Data	<-	TraitementAge(Data)
Data_submit<-Cleaning(Data_submit)
Data_submit<-TraitementAge(Data_submit)
preds_matrix<<-data.frame(nrow=nrow(Data_submit),ncol=2)


#profile1

profile1	<-list(
			"sex"=c("female","male"),
			"age"=c(0,90),
			"classe"=c(1,2,3),
			"parents"=c(0,10),
			"embarked"=c("S","Q","C"),
			"siblings"=c(0,10),
			"view"=FALSE,
			"Methods"=c('randomForest','naiveBayes','ksvm','AdaBoost','mlp'))
profile2	<-list(
			"sex"=c("female","male"),
			"age"=c(0,90),
			"classe"=c(2,3),
			"parents"=c(0,10),
			"embarked"=c("S","Q","C"),
			"siblings"=c(0,10),
			"view"=FALSE,
			"Methods"=c('randomForest','naiveBayes','ksvm','AdaBoost','mlp'))
profile3	<-list(
			"sex"=c("female"),
			"age"=c(0,90),
			"classe"=c(1,2),
			"parents"=c(0,10),
			"embarked"=c("S","Q","C"),
			"siblings"=c(0,10),
			"view"=FALSE,
			"Methods"=c('randomForest','naiveBayes','ksvm','AdaBoost','mlp'))

profile4	<-list(
			"sex"=c("female"),
			"age"=c(0,90),
			"classe"=c(2,3),
			"parents"=c(0,10),
			"embarked"=c("S","Q","C"),
			"siblings"=c(0,10),
			"view"=FALSE,
			"Methods"=c('randomForest','naiveBayes','ksvm','AdaBoost','mlp'))

PredFromProfile( Train=Data, Test=Data_submit, Profile=profile1)
PredFromProfile( Train=Data, Test=Data_submit, Profile=profile2)
PredFromProfile( Train=Data, Test=Data_submit, Profile=profile3)
PredFromProfile( Train=Data, Test=Data_submit, Profile=profile4)


preds_matrix[,1]<-preds_matrix[,1]+N_data
preds_matrix[,2]<-factor(preds_matrix[,2],labels=c(0,1));preds_matrix[,2]<-as.numeric(as.character(preds_matrix[,2]));colnames(preds_matrix)<-c("PassengerId","Survived")
write.table(preds_matrix,paste(output.path,"Submission",".csv",sep=""),row.names=FALSE,sep=",")
View(preds_matrix)























###########################################
#################Stuff#####################
###########################################
########
###PCA##
########
#####
#acf<- princomp(Data[,!(names(Data_test) %in% c('Survived'))], cor=TRUE)
#summary(acf) #print la variance expliquee
#loadings(acf) # pc loadings 
#plot(acf,type="lines") # scree plot 
#View(acf$scores) # the principal components
#biplot(acf)
#Xpca=acf$scores[,1:3]
#Data<-data.frame("Survived"=Data[,'Survived'],Xpca)
#View(data.frame("Survived"=Data[,'Survived'],Xpca))
#####
preds1<-as.numeric(preds1)
preds2<-as.numeric(preds2)
preds3<-as.numeric(preds3);preds3[which(preds3==1)]=2;preds3[which(preds3==0)]=1;
preds4<-as.numeric(preds4)
preds5<-as.numeric(preds5)
preds6<-as.numeric(preds6)
preds8<-as.numeric(preds8)

#il y aurait moyen d'optimiser ici en faisant une somme pondérée et en utilisant une bibli d'optimisation pour touver les meilleurs poids  
moy=(preds1+preds2+preds3+preds4+preds5+preds6+preds8)/7
preds<-NULL;
preds[moy>1.5]=1;preds[moy<=1.5]=0
score(Preds=preds,True=Data_test[,'Survived'])

