backward.selection<-function(data,y,folds=10){  
  model <- lm(y ~ ., data = cbind(data,y))
  #train.x <- data.matrix(data)
  #train.y <- data.matrix(y)
  #model <- glmnet(train.x,train.y,family="gaussian")
  #step() function only accepts lm() & glm() not glmnet()
  back.elim <- step(model,direction="backward",trace=F)
  return(back.elim)
}

get.poly<-function(dat,degree){
  poly<-data.frame(lapply(dat,poly,data=degree,raw=T))
}

lm_prediction <- function(lm.f,test){
  names(test)<-c('apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm','spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m', 'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sf') 
  sc.test<-data.frame(scale(test))
  testpoly3<-get.poly(sc.test,3)
  pred<-predict(lm.f,testpoly3)
  #pred<-predict(lm.f,type="response",newx=testpoly3)
  submat<-data.frame(matrix(pred,ncol=98,byrow=T))
  return(submat)
}

get.station_names<-function(f){
  st.names<-read.csv(f,nrow=1) 
  st.names<-names(st.names)[-1]
}

readData<-function(fname='spline_train500.csv'){
  data = read.csv(fname,header=F)
  dat = data[,c(2:16)]
  target = data[,17]
  names(dat)<-c('apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm','spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m', 'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sf')
  return(list(dat = dat, target = target))
}

conclude<-function(datList,degree){
  dat<-datList$dat
  target<-data.frame(y=datList$target)
  scdat<-data.frame(scale(dat))
  poly3<-get.poly(scdat,degree)
  model<-backward.selection(poly3,target)
}
#test<-read.csv("C:/Users/Administrator/Documents/Java_Proj/eclipse/python/solar/src/sol/spline_test.csv",header=F)
test.model<-function(test,model){
  sub<-lm_prediction(model,test)
  return(sub)
}

#write.submission("E:/kaggle/Solar/gefs_test/test/sampleSubmission.csv","C:/Users/Administrator/Documents/Java_Proj/eclipse/python/solar/src/sol/back.elim.submission.csv",tm)
write.submission<-function(infn,outfn,submat){
  samplefile<-read.csv(infn)
  sf<-samplefile[,1]
  names(submat)<-get.station_names(infn)
  sub<-cbind(Date=sf,submat)
  write.csv(sub,outfn,row.names=F)
}