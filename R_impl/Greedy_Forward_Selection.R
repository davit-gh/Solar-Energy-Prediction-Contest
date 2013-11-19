readData<-function(fname='spline_train500.csv'){
  data = read.csv(fname,header=F)
  dat = data[,c(2:16)]
  target = data[,17]
  names(dat)<-c('apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm','spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m', 'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sf')
  return(list(dat = dat, target = target))
}

nfoldcv3<-function(data,n){
  id<-sample(rep(seq_len(n),length.out=nrow(data)))
  outList<-lapply(seq_len(n),function(x){
    train = data[id != x,]
    test = data[id == x,]
    return(list(
      train = train,
      test = test
    ))
  })
  return(outList)
}

cv3<-function(data,n){
  out <- nfoldcv3(data,n)
  maeList <- lapply(out, function(x){
    #lm.y <- lm(y ~ ., data = x$train)
    train.x <- data.matrix(subset(x$train,select=-y))
    train.y <- data.matrix(x$train['y'])
    
    lm.y <- glmnet(train.x,train.y,family="gaussian")
    test.y <- x$test['y']
    test.x <- data.matrix(subset(x$test,select=-y))
    pred <- predict(lm.y,type="response",newx=test.x)
    MAE <- mae(test.y,pred)
    return(MAE)
  })
  meanMAE <- sum(unlist(maeList))/n
  return(meanMAE)                  
}

lm_prediction6<-function(lm.f,test){
  names(test)<-c('apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm','spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m', 'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sf') 
  sc.test<-data.frame(scale(test))
  testpoly3<-get.poly(sc.test,3)
  #pred<-predict(lm.f,testpoly3)
  pred<-predict(lm.f,type="response",newx=testpoly3)
  submat<-data.frame(matrix(pred,ncol=98,byrow=T))
  return(submat)
}
select.model<-function(){
  y<-data[,16]
  data<-data[,-16]
  data1<-sweep(data,2,colMeans(data),"-")
  data1<-sweep(data1,2,sapply(data1,sd),"/")
  y<-data.frame(y=y)
 
f<-function(data){  
  selected = list()
  nr <- nrow(data)
  while(length(data) != nr){
    if(length(selected) > 0){
      newdata<-lapply(data,cbind,selected,y)
    }
    else{
      newdata<-lapply(data,cbind,y)      
    }
    res<-lapply(newdata,cv2,5)
    min.index <- which.min(res)
    cat(names(min.index),res[[min.index]],'\n')
    selected <- c(selected,subset(data,select=min.index))
    data <- data[,-min.index]
  }
  return(selected)
}
last <- function(x,n=0){
  x[[length(x)-n]]
}
  
  
 f.upd3<-function(data,y,folds=10){  
    selected <- list()
    score_hist <- list()
    while(length(score_hist) < 2 || last(score_hist) < last(score_hist,1)){
      if(length(selected) > 0){
        newdata<-lapply(data,cbind,selected,y=y)
      }
      else{
        newdata<-lapply(data,cbind,y=y)      
      }
      res<-lapply(newdata,cv3,folds)
      min.index <- which.min(res)
      score_hist<-c(score_hist,res[[min.index]])
      cat(names(min.index),res[[min.index]],'\n')
      selected <- c(selected,subset(data,select=min.index))
      data <- data[,-min.index]
    }
    return(list(selected=head(selected,-1),score_hist=score_hist))
  }
  

get.poly<-function(dat,degree){
 poly<-data.frame(lapply(dat,poly,data=degree,raw=T))
}
get.station_names<-function(f){
 st.names<-read.csv(f,nrow=1) 
 st.names<-names(st.names)[-1]
}

conclude<-function(){
  datList<-readData()
  dat<-datList$dat
  target<-datList$target
  scdat<-data.frame(scale(dat))
  sc.y<-mean.sigma.y(target)$scaled.y
  sc.y<-data.frame(y=sc.y)
  poly3<-get.poly(scdat,3)
  s.updx<-f.upd3(poly3,sc.y)
}


conclude2<-function(datList,polyDeg){
  dat<-datList$dat
  target<-data.frame(y=datList$target)
  scdat<-data.frame(scale(dat))
  poly3<-get.poly(scdat,polyDeg)
  s.updx<-f.upd3(poly3,target)
}

test.model2<-function(data,test,s.updx){
  trdat<-data.frame(s.updx$selected)
  y<-data.frame(y=data$target)
  finaltrainingdata<-cbind(trdat,y)
  model<-lm(y~.,data=finaltrainingdata)
  sub<-lm_prediction6(model,test)
  return(sub)
}
write.submission<-function(infn,outfn,submat){
  samplefile<-read.csv(infn)
  sf<-samplefile[,1]
  names(submat)<-get.station_names(infn)
  sub<-cbind(Date=sf,submat)
  write.csv(sub,outfn,row.names=F)
}
