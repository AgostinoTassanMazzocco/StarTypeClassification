library(readr)
library(caret)
library(car)
library(class)
library(MASS)
library(ggcorrplot)
library(heplots)
require(gridExtra)
require(e1071)
require(kernlab)
require(mlrMBO)
require(GGally)
require(tree)
require(randomForest)

Stars <- read_csv("C:/Users/Lenovo/OneDrive/Desktop/Università/Magistrale/Machine Learning/Progetto/Stars.csv")

summary(Stars)
anyNA(Stars) #False
for(i in 0:5){                        #dataset bilanciato
  print(length(which(Stars$Type==i)))
}

for(i in 1:nrow(Stars)){
  if(Stars$Color[i]=='Blue-white' | Stars$Color[i]=='Blue-White' | Stars$Color[i]=='Blue white' | 
     Stars$Color[i]=='Blue White'){Stars$Color[i]='Blue-white'}
  if(Stars$Color[i]=='white' | Stars$Color[i]=='White' | Stars$Color[i]=='Whitish'){Stars$Color[i]='White'}
  if(Stars$Color[i]=='Orange' | Stars$Color[i]=='orange' | Stars$Color[i]=='Pale yellow orange' |
     Stars$Color[i]=='Orange-Red'){Stars$Color[i]='Orange'}
  if(Stars$Color[i]=='White-Yellow' | Stars$Color[i]=='yellow-white') {Stars$Color[i]="Yellow-white"}
  if(Stars$Color[i]=='yellowish' | Stars$Color[i]=='Yellowish' | 
     Stars$Color[i]=='Yellowish White'){Stars$Color[i]='Yellow'}
}

for(i in 1:nrow(Stars)){
  if(Stars$Color[i]=='White'){Stars$Color[i]=5}
  if(Stars$Color[i]=='Blue-white'){Stars$Color[i]=6}
  if(Stars$Color[i]=='Blue'){Stars$Color[i]=7}
  if(Stars$Color[i]=='Orange'){Stars$Color[i]=2}
  if(Stars$Color[i]=='Red'){Stars$Color[i]=1}
  if(Stars$Color[i]=='Yellow'){Stars$Color[i]=3}
  if(Stars$Color[i]=="Yellow-white"){Stars$Color[i]=4}
  if(Stars$Spectral_Class[i]=='M'){Stars$Spectral_Class[i]=1}
  if(Stars$Spectral_Class[i]=='K'){Stars$Spectral_Class[i]=2}
  if(Stars$Spectral_Class[i]=='G'){Stars$Spectral_Class[i]=3}
  if(Stars$Spectral_Class[i]=='F'){Stars$Spectral_Class[i]=4}
  if(Stars$Spectral_Class[i]=='A'){Stars$Spectral_Class[i]=5}
  if(Stars$Spectral_Class[i]=='B'){Stars$Spectral_Class[i]=6}
  if(Stars$Spectral_Class[i]=='O'){Stars$Spectral_Class[i]=7}
}

Stars$Spectral_Class=as.numeric(Stars$Spectral_Class)
Stars$Color=as.numeric(Stars$Color)
Stars$Type=as.factor(Stars$Type)

#############################################################################################################
################################ ANALISI ESPLORATIVA GRAFICA ################################################
#############################################################################################################

ggpairs(Stars[,-7], aes(colour=as.factor(Stars$Type)))
table(Stars$Type,Stars$Spectral_Class)
table(Stars$Type,Stars$Color)
#Sarebbe opportuno non considerare Color e Spectral Class che sono molto correlate tra loro e con Temperature

ggplot(Stars, aes_string(x='A_M', y='Temperature', colour='Type'))+
  geom_point()+
  scale_color_manual(aesthetics = rainbow(length(levels(as.factor(Stars$Type)))))+
  ggtitle(label = 'Scatterplot', subtitle = 'rispetto a temperatura e AM')+
  theme(panel.grid = element_line(colour = 'grey', linetype = 3), panel.background = element_rect('white'),
        text = element_text(face = 'bold.italic'), axis.line = element_line(colour = 'grey15', linetype = 1),
        legend.background = element_rect(fill = 'lightgrey'))

ggcorrplot(corr= cor(Stars[,-7]), method = 'circle', type = 'lower', title = 'Correlation Plot',
           show.diag = F, colors = c('darkblue', 'green', 'red'), lab = T, lab_col = 'grey15', 
           outline.color =  'white', lab_size = 4, digits = 1, hc.order = T, hc.method = 'complete')

plot_box_all <-list()
var<- colnames(Stars[,-7])   #Costruzione dei boxplot condizionati
for (i in var){
  plot_box_all[[i]] <- ggplot(Stars, aes_string(x = "Type", y = i, col = "Type", fill = "Type")) + 
    geom_boxplot(alpha = 0.6, outlier.size = 2) + 
    theme(legend.position = "none", plot.background = element_rect('grey15'),
          text = element_text(face = 'bold.italic', color = 'white'),
          panel.background = element_rect('white'), panel.grid = element_line(colour = 'grey', linetype = 3))+
    scale_color_discrete(rainbow(n=length(var)))
}
do.call(grid.arrange, c(plot_box_all, nrow = 3))

#Grafico sulla distribuzione delle variabili esplicative
covEllipses(Stars[,-7], 
            factor(Stars$Type), 
            fill = TRUE,       
            pooled = FALSE,    
            col = c(1:length(var)), 
            variables = c(1:length(var)), 
            fill.alpha = 0.1)

############################################################################################################

set.seed(456)
stars_scaled=cbind(scale(Stars[,-c(5,6,7)]), Stars[,7])
tr_ind=sample(nrow(Stars), size = floor(0.85*nrow(Stars)), replace = F)
tr_set_scaled=stars_scaled[tr_ind,]
colnames(tr_set_scaled)=c("Temperature", "L", "R", "A_M", "Type")
val_set_scaled=tr_set_scaled[1:24,]
test_set_scaled=stars_scaled[-tr_ind,]
for(i in 0:5){                        
  print(length(which(tr_set_scaled$Type==i)))
}

calc.class.err = function(actual, predicted) {
  mean(actual != predicted)
}

############################################################################################################
####################################### CLASSIFICAZIONE ####################################################
############################################################################################################

#KNN
k=c(1,3,5,7,9)
acc=vector()
for(j in 1:length(k)){
  cv.knn=knn.cv(train = tr_set_scaled[,-5], cl = tr_set_scaled[,5], k = k[j]) #leave-one-out su training set
  cm=confusionMatrix(as.factor(cv.knn), as.factor(tr_set_scaled[,5]))$table
  acc[j]=sum(diag(cm))/length(tr_set_scaled[,5])
}
acc
plot(k, acc, type = "b", col = "dodgerblue", cex = 1, pch = 20, xlab = "k, number of neighbours", 
     ylab = "classification accuracy", main = "Training Accuracy Rate vs Neighbours")
abline(h = acc[2], col = "darkorange", lty = 3)
#La scelta di k=3 è dovuta al fatto che il Knn con k=1 è spesso fonte di Overfitting. 
mod.knn=knn.cv(train = tr_set_scaled[,-5], cl = tr_set_scaled[,5], k = 3) #leave-one-out su training set
(confusionMatrix(as.factor(mod.knn), as.factor(tr_set_scaled[,5]))) #2 errori
tr_set_scaled[which(tr_set_scaled[,5]!=as.factor(mod.knn)),]  #50,101

#SVM lineare
acc=c()
err=c()
cost=seq(1, 3, length.out = 20)
for(i in 1:length(cost)){
  mod_svm_lin=svm(tr_set_scaled[-c(1:24),-5], tr_set_scaled[-c(1:24),5], data=tr_set_scaled[-c(1:24),],
                  scale=F, type='C-classification', kernel='linear', cost = cost[i])
  c=confusionMatrix(as.factor(mod_svm_lin$fitted), as.factor(tr_set_scaled[-c(1:24),5]))$table
  acc[i]=sum(diag(c))/nrow(tr_set_scaled[-c(1:24),])
  fit_svm=predict(object = mod_svm_lin, newdata = as.matrix(val_set_scaled[,-5]))
  err[i]=calc.class.err(as.factor(fit_svm), as.factor(val_set_scaled[,5]))
}
cost[which.max(acc)] #cost=1.53 è scelto come miglior iperparametro e i dati risultano linearmente separabili
acc;err   #errore calcolato su validation set

acc=c()
err=c()
for(h in 1:nrow(tr_set_scaled)){
  tr.s=tr_set_scaled[-h,]
  val.s=tr_set_scaled[h,]
  mod_svm_lin=svm(tr.s[,-5], tr.s[,5], data=tr.s, scale=F, type='C-classification', kernel='linear', cost = 1.53)
  c=confusionMatrix(as.factor(mod_svm_lin$fitted), as.factor(tr.s[,5]))$table
  acc[h]=sum(diag(c))/nrow(tr.s)
  fit_svm=predict(object = mod_svm_lin, newdata = data.frame(val.s[,-5]))
  err[h]=calc.class.err(as.factor(fit_svm), as.factor(val.s[,5]))
  if(err[h]>0){print(h);print(fit_svm)}
}
mean(acc);sum(err)
tr_set_scaled[which(err>0),]   #50,103


#SVM con kernel radiale
cost=seq(1, 3, length.out = 20)
gam=seq(0.7, 6.3, length.out = 20)
acc=matrix(NA, nrow = length(cost), ncol = length(gam))
err=matrix(NA, nrow = length(cost), ncol = length(gam))
for(i in 1:length(cost)){    #richiede qualche secondo
  for(j in 1:length(gam)){
    mod_svm_ker=svm(tr_set_scaled[-c(1:24),-5], tr_set_scaled[-c(1:24),5], data=tr_set_scaled[-c(1:24),],
                    scale=F, type='C-classification', kernel='radial', cost = cost[i], gamma = gam[j])
    c=confusionMatrix(as.factor(mod_svm_ker$fitted), as.factor(tr_set_scaled[-c(1:24),5]))$table
    acc[i,j]=sum(diag(c))/nrow(tr_set_scaled[-c(1:24),])
    fit_svm=predict(object = mod_svm_ker, newdata = as.matrix(val_set_scaled[,-5]))
    err[i,j]=calc.class.err(as.factor(fit_svm), as.factor(val_set_scaled[,5]))
  }
}
acc;err
gam[6];cost[13]

acc=c()
err=c()
for(h in 1:nrow(tr_set_scaled)){
  tr.s=tr_set_scaled[-h,]
  val.s=tr_set_scaled[h,]
  mod_svm_ker=svm(tr.s[,-5], tr.s[,5], data=tr.s, scale=F, type='C-classification', 
                  kernel='radial', cost = cost[13], gamma = gam[6])
  c=confusionMatrix(as.factor(mod_svm_ker$fitted), as.factor(tr.s[,5]))$table
  acc[h]=sum(diag(c))/nrow(tr.s)
  fit_svm=predict(object = mod_svm_ker, newdata = data.frame(val.s[,-5]))
  err[h]=calc.class.err(as.factor(fit_svm), as.factor(val.s[,5]))
}
mean(acc);sum(err)
tr_set_scaled[which(err==1),] #50,224,228


#SVM polynomial
cost=seq(1, 6, length.out = 20)
gam=seq(0.2, 10, length.out = 20)
deg=c(2,3,4,5,6)
A=list()
E=list()
acc=matrix(NA, nrow = length(cost), ncol = length(gam))
err=matrix(NA, nrow = length(cost), ncol = length(gam))

for(k in 1:length(deg)){       #richiede qualche secondo in più
  for(i in 1:length(cost)){
    for(j in 1:length(gam)){
      mod_svm_pol=svm(tr_set_scaled[-c(1:24),-5], tr_set_scaled[-c(1:24),5], data=tr_set_scaled[-c(1:24),], 
                      scale=F, type='C-classification', kernel='polynomial', cost = cost[i], gamma = gam[j], 
                      degree = deg[k])
      c=confusionMatrix(as.factor(mod_svm_pol$fitted), as.factor(tr_set_scaled[-c(1:24),5]))$table
      acc[i,j]=sum(diag(c))/nrow(tr_set_scaled[-c(1:24),])
      fit_svm=predict(object = mod_svm_pol, newdata = as.matrix(val_set_scaled[,-5]))
      err[i,j]=calc.class.err(as.factor(fit_svm), as.factor(val_set_scaled[,5]))
    }
  }
  A[[k]]=acc
  E[[k]]=err
  acc=matrix(NA, nrow = length(cost), ncol = length(gam))
  err=matrix(NA, nrow = length(cost), ncol = length(gam))
}     
A #degree=2, gamma=0.7157895, cost=1  Al crescere del grado polinomiale si ha meno precisione nel tr.set
E #Nessun errore nel validation set per questi iperparaetri

acc=c()
err=c()
for(h in 1:nrow(tr_set_scaled)){
  tr.s=tr_set_scaled[-h,]
  val.s=tr_set_scaled[h,]
  mod_svm_pol=svm(tr.s[,-5], tr.s[,5], data=tr.s, scale=F, type='C-classification', 
                  kernel='polynomial', cost = 1, gamma = 0.8, degree = 2)
  c=confusionMatrix(as.factor(mod_svm_pol$fitted), as.factor(tr.s[,5]))$table
  acc[h]=sum(diag(c))/nrow(tr.s)
  fit_svm=predict(object = mod_svm_pol, newdata = data.frame(val.s[,-5]))
  err[h]=calc.class.err(as.factor(fit_svm), as.factor(val.s[,5]))
}
mean(acc);sum(err) #Fit perfetto
mod_svm_pol=svm(tr_set_scaled[,-5], tr_set_scaled[,5], data=tr_set_scaled, scale=F, type='C-classification', 
                kernel='polynomial', cost = 1, gamma = 0.8, degree = 2)
summary(mod_svm_pol) #61 SV

#Confronto grafico
par(mfrow=c(2,2))
plot(tr_set_scaled[-c(134,146),4]~tr_set_scaled[-c(134,146),1], pch=19, col=tr_set_scaled[-c(134,146),5], 
     main='Fitted values by Linear SVM', xlab='Temperatura', ylab='Magnitudine Assoluta')
points(tr_set_scaled[c(134,146),4]~tr_set_scaled[c(134,146),1], pch=9, col=4)

plot(tr_set_scaled[-c(83,134,156),4]~tr_set_scaled[-c(83,134,156),1], pch=19, col=tr_set_scaled[-c(83,134,156),5], 
     main='Fitted values by Radial SVM', xlab='Temperatura', ylab='Magnitudine Assoluta')
points(tr_set_scaled[c(83,134,156),4]~tr_set_scaled[c(83,134,156),1], pch=9, col=4)

plot(tr_set_scaled[,4]~tr_set_scaled[,1], pch=19, col=tr_set_scaled[,5], 
     main='Fitted values by Polynomial SVM', xlab='Temperatura', ylab='Magnitudine Assoluta')

plot(tr_set_scaled[-c(134,102),4]~tr_set_scaled[-c(134,102),1], pch=19, col=tr_set_scaled[-c(134,102),5], 
     main='Fitted values by 3-NN', xlab='Temperatura', ylab='Magnitudine Assoluta')
points(tr_set_scaled[c(134,102),4]~tr_set_scaled[c(134,102),1], pch=9, col=4)

par(mfrow=c(1,1))

####################################################################################################
################################## AUTO MACHINE LEARNING ###########################################
####################################################################################################

#SVM
set.seed(411)
par.set = makeParamSet(
  makeDiscreteParam( "kernel", values=c("linear", "radial", 'polynomial')),
  makeNumericParam( "cost", lower=-2, upper=2, trafo=function(x) 10^x ),
  makeNumericParam( "gamma", lower=-2, upper=2, trafo=function(x) 10^x, requires=quote(kernel!="linear")),
  makeDiscreteParam( "degree", values=c(2, 3, 4, 5, 6), requires=quote(kernel=="polynomial"))
)

ctrl = makeMBOControl()
ctrl = setMBOControlTermination(ctrl, max.evals=20 )
tune.ctrl = makeTuneControlMBO(mbo.control = ctrl)
task <- makeClassifTask(data=tr_set_scaled, target='Type')

run = tuneParams(makeLearner("classif.svm"), task, cv3, par.set = par.set, control = tune.ctrl, show.info = T)

sequential.err <- getOptPathY(run$opt.path)
best.seen <- cummin(sequential.err) 
plot( sequential.err, type="o", lwd=3, col="green", lty=2 )
lines( best.seen, type="o", col="blue", lwd=3 )
run$mbo.result #kernel=linear; cost=1.32

mod_svm_best=svm(tr_set_scaled[-c(1:24),-5], tr_set_scaled[-c(1:24),5], data=tr_set_scaled[-c(1:24),],
                 scale=F, type='C-classification', kernel='linear', cost = 1.32)
summary(mod_svm_best)
(confusionMatrix(as.factor(mod_svm_best$fitted), as.factor(tr_set_scaled[-c(1:24),5]))) #non soddisfacente
fit_svm_b=predict(object = mod_svm_best, newdata = val_set_scaled[,-5])
(confusionMatrix(as.factor(fit_svm_b), as.factor(val_set_scaled[,5])))

#########################################################################################################

#Tree con 2 variabili 
err=c()
for(h in 1:nrow(tr_set_scaled)){
  tr.s=tr_set_scaled[-h,c(1,4,5)]
  val.s=tr_set_scaled[h,c(1,4,5)]
  tr.mod=tree(factor(Type)~Temperature+A_M, data=data.frame(tr.s), split = 'deviance',
              control = tree.control(nobs = nrow(tr.s), mincut = 5, minsize = 10, mindev = 0.01))
  fit.tree=predict(newdata = data.frame(val.s), object = tr.mod)
  fit.tree=apply(fit.tree, MARGIN = 1, FUN = function(x) which.max(x)-1)
  err[h]=calc.class.err(as.factor(fit.tree), as.factor(val.s[,3]))
}
which(err==1) #Fit perfetto

tr.mod=tree(factor(Type)~Temperature+A_M, data=data.frame(tr_set_scaled), split = 'deviance',
            control = tree.control(nobs = nrow(tr_set_scaled), mincut = 5, minsize = 10, mindev = 0.01))
par(mfrow=c(1,2))
plot(tr.mod);text(tr.mod)
title(main = 'Tree plot')
partition.tree(tr.mod, main='Partition Tree')
par(mfrow=c(1,1))

#Grafico
gx=seq(min(tr_set_scaled[,1],tr_set_scaled[,4]), max(tr_set_scaled[,1], tr_set_scaled[,4]), length.out = 200)
grid=expand.grid(gx, gx)
pred.grid <- predict(object = tr.mod, newdata=data.frame(Temperature=grid[,1], A_M=grid[,2]))
pred.grid=apply(pred.grid, MARGIN = 1, FUN = function(x) which.max(x))
plot(tr_set_scaled[,4]~tr_set_scaled[,1], pch=19, cex=1.3, col=tr_set_scaled[,5], main='Classification Tree',
     xlab='Temperature scaled', ylab='Absolute Magnitude scaled')
points(grid, col=pred.grid, pch=19, cex=0.2)

#RandomForest
(rf.mod=randomForest(x=tr_set_scaled[-c(1:24),-5], y=factor(data.frame(tr_set_scaled)[-c(1:24),5]), 
                     data=tr_set_scaled[-c(1:24),], ntree = 30, mtry = 2))
fit.rf=predict(rf.mod, newdata=val_set_scaled[,-5])
(confusionMatrix(as.factor(fit.rf), as.factor(val_set_scaled[,5]))) #Ovviamente fit perfetto come per albero
plot(rf.mod)
summary(rf.mod)


par(mfrow=c(1,2))

#lda
mod_lda=lda(Type~Temperature+A_M, data = tr_set_scaled)
#Grafico
gx=seq(min(tr_set_scaled[,1],tr_set_scaled[,4]), max(tr_set_scaled[,1], tr_set_scaled[,4]), length.out = 200)
grid=expand.grid(gx, gx)
pred.grid <- predict(object = mod_lda, newdata=data.frame(Temperature=grid[,1], A_M=grid[,2]))
plot(tr_set_scaled[,4]~tr_set_scaled[,1], pch=19, col=tr_set_scaled[,5], main='Classification LDA',
     xlab='Temperature scaled', ylab='Absolute Magnitude scaled')
points(grid, col=pred.grid$class, pch=19, cex=0.1)

#qda
mod_qda=qda(Type~Temperature+A_M, data = tr_set_scaled)
#Grafico
gx=seq(min(tr_set_scaled[,1],tr_set_scaled[,4]), max(tr_set_scaled[,1], tr_set_scaled[,4]), length.out = 200)
grid=expand.grid(gx, gx)
pred.grid <- predict(object = mod_qda, newdata=data.frame(Temperature=grid[,1], A_M=grid[,2]))
plot(tr_set_scaled[,4]~tr_set_scaled[,1], pch=19, col=tr_set_scaled[,5], main='Classification QDA',
     xlab='Temperature scaled', ylab='Absolute Magnitude scaled')
points(grid, col=pred.grid$class, pch=19, cex=0.1)

par(mfrow=c(1,1))

############################################################################################################
################################## MODELLO FINALE SU TEST SET ##############################################
############################################################################################################

#Il modello con maggiore performance e minore complessità computazionale è l'albero decisionale 
err=c()
final.mod=tree(factor(Type)~Temperature+A_M, data=data.frame(tr_set_scaled))
fit.tree=predict(newdata = data.frame(test_set_scaled), object = final.mod)
fit.tree=apply(fit.tree, MARGIN = 1, FUN = function(x) which.max(x)-1)
err=calc.class.err(as.factor(fit.tree), as.factor(test_set_scaled[,5]))
plot(final.mod);text(final.mod)
err #Non ci sono errori nel test set
confusionMatrix(as.factor(fit.tree), as.factor(test_set_scaled[,5])) 

gx=seq(min(stars_scaled[,1],stars_scaled[,4]), max(stars_scaled[,1], stars_scaled[,4]), length.out = 110)
grid=expand.grid(gx, gx)
pred.grid <- predict(object = final.mod, newdata=data.frame(Temperature=grid[,1], A_M=grid[,2]))
pred.grid=apply(pred.grid, MARGIN = 1, FUN = function(x) which.max(x))
plot(test_set_scaled[,4]~test_set_scaled[,1], pch=19, cex=1.2, col=test_set_scaled[,5], 
     xlab='Temperature_scaled', ylab='A_M_scaled', main='Test Set Classification Tree')
points(grid, col=pred.grid, pch=19, cex=0.2)

