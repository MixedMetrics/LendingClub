import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

file_date = "Jun-2016"


print "\n\n\nMixed Metrics for Lending Club...\n\n" 
print "reading the csv \n"

train = pd.read_csv('train.csv', header = 1)  #there's an extra line before the column names, thus header =1 
loans =  len(train)
train = train.head(loans-2) # - there was a couple extra lines at the end of the file that we don't need
print "processing", loans, "entries total....  \n"

print "dropping low grade loans...\n"   #doing this because recent loans are higher grade than in past. 

train = pd.get_dummies(train, columns = ['grade'])
high_grade=train.grade_A+train.grade_B+train.grade_C
train = train[(high_grade ==1)]

# first we compute our target value.  Since this computation is different for each status, we use dummies to vectorize this calculation.  My experience is the pulling the columns out of the dataframe and handling them with numpy is much quicker than trying to work in the dataframe.  
print "\n computing target value"  
assumed_interest_rate = 0.07
r = assumed_interest_rate  #just a number. I got about 8% via hunt and peck a few years ago.  The idea here is this, if you got this loan and continually invested the proceeds of the loan back into something that returns 7%, how much money would that get you in 5 years?  This is biased against more recent loans, but I don't think that will hurt the predictions much. 

train = pd.get_dummies(train, columns = ['loan_status'])
today = pd.to_datetime(file_date)
funded = train.funded_amnt
principle = train.out_prncp
received = train.total_pymnt
value = principle+received

train['issue_d'].fillna(file_date, inplace=True)
issue_d = pd.to_datetime(train.issue_d)

train['last_pymnt_d'].fillna(file_date,inplace=True)
last_pymnt_d=pd.to_datetime(train.last_pymnt_d)

length = list(last_pymnt_d -issue_d)
length = [float(i.days)+1 for i in length]   ## I added the +1 here because a couple of loans where paid off in the same month they were issued, which meant getting time delta zero
length = np.array(length)/365.0  #days to year

payment_rate = received/length
reinvestment_balance = (np.exp(r*length)-1)*payment_rate/r
try: reinvestment_balance += train.loan_status_Current*principle +train['loan_status_Late (16-30 days)']*0.42*principle+train['loan_status_Late (31-120 days)']*0.26*principle+train['loan_status_In Grace Period']*0.72*principle   #I'm using a approximation for the recovery percentage, that Lending Club publishes.  
except: reinvestment_balance += train.loan_status_Current*principle   #throws an error if the file is small and one of these features has not been created
 

after_five_years = reinvestment_balance*np.exp(r*(5-length))
ratio = after_five_years/funded
print ratio
target = ratio
print "\n target computed.. now drop some columns"
 
to_drop = ['id','member_id','funded_amnt','funded_amnt_inv','pymnt_plan','url','desc','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d'
,'last_fico_range_high','last_fico_range_low','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d','last_pymnt_d','issue_d']   #not completely necessary to do this, but these columns annoyed me. 


train.drop(to_drop, axis=1, inplace=True)
train.drop('verification_status_joint',axis=1, inplace=True)  # this didn't seem worth processing
train['emp_title'].fillna('notgiven', inplace = True) 
train.fillna(-1.0,inplace=True, downcast = 'infer')


train['int_rate'] = train['int_rate'].replace('%','',regex=True).astype('float')
train['revol_util'] = train['revol_util'].replace('%','',regex=True).astype('float')
ecl=list(pd.to_datetime(train.earliest_cr_line)-today)
ecl = [int(i.days) for i in ecl]

train['earliest_cr_line']=ecl


train['term'] = train['term'].replace('months','',regex=True).astype('int')
train['zip_code'] = train['zip_code'].replace('xx','',regex=True).astype('float')


print "\n... creating dummy columns\n"

train['emp_length'] = train['emp_length'].replace('<','lessthan',regex=True)
to_dummy = ['initial_list_status','term','sub_grade','application_type','title','home_ownership', 'purpose','addr_state','emp_length','verification_status']
train = pd.get_dummies(train, columns = to_dummy)

print "finding most common words in employment title"

from collections import Counter
NE = 40  #number of words in employment title

emplist = list(train.emp_title)
emplist = [x.lower() for x in emplist]   #make all words lowercase 
wl1 = [i.split() for i in emplist]
flattened = [i for sublist in wl1 for i in sublist]
counts = Counter(flattened)
wl2 = [x for x,y in counts.most_common(NE)]

print wl2
print "\nmaking columns for these words"


mm = np.zeros((len(emplist),NE))
i =0
for emp in wl1:
    j = 0
    for word in wl2:
        if word in emp:
            mm[i][j]=1
        j += 1 
    i+=1 

i=0
for x in wl2:
    train[x]=mm[:,i]
    i+=1



train.drop('emp_title',axis=1, inplace=True)








print "\nTrain file done, now opening test file"
test = pd.read_csv("retail.csv")
test_id = test.id
print "\nprocessing test file..."


test['verification_status']=test['is_inc_v']
test['emp_title'].fillna('notgiven', inplace = True) 
test.fillna(-1.0,inplace=True, downcast = 'infer')

test['mths_since_recent_bc'].fillna('500', inplace = True)
problem_columns = ['mths_since_recent_bc','dti_joint','mths_since_last_delinq','num_tl_120dpd_2m','bc_util','percent_bc_gt_75',
'mths_since_recent_bc_dlq','mths_since_recent_inq','il_util','mths_since_last_record','mths_since_recent_revol_delinq','annual_inc_joint','bc_open_to_buy']
## I'm not sure why, but these columns are read into pandas as lists.  This makes them resistant to fillna.   So I'm force-filling the NA.  

print "\ndealing with the annoying list columns"


templist =np.zeros(len(test))
for col in problem_columns:
    a = list(test[col])
    i=0
    for k in a:
        try : k = int(k)
        except: k = 500
        templist[i]=k
        i+=1
    test[col] = templist



ecl=list(pd.to_datetime(test.earliest_cr_line)-today)
ecl = [int(i.days) for i in ecl]

test['earliest_cr_line']=ecl
test['zip_code'] = test['zip_code'].replace('xx','',regex=True).astype('float')
test['emp_length'] = test['emp_length'].replace('<','lessthan',regex=True)

test['mths_since_last_major_derog'] = test['mths_since_last_major_derog'].replace('null','-1',regex=True).astype('int')
to_dummy = ['initial_list_status','term','sub_grade','application_type','title','home_ownership', 'purpose','addr_state','emp_length','verification_status']
test = pd.get_dummies(test, columns = to_dummy)


print "finding most common words in employment title"


emplist_test = list(test.emp_title)
emplist_test = [x.lower() for x in emplist_test]   #make all words lowercase 
wl1_test = [i.split() for i in emplist_test]
flattened_test = [i for sublist in wl1_test for i in sublist]
counts_test = Counter(flattened)
wl2_test = [x for x,y in counts_test.most_common(NE)]

print wl2_test
print "\nmaking columns for these words"


mm_test = np.zeros((len(emplist_test),NE))
i =0
for emp in wl1_test:
    j = 0
    for word in wl2_test:
        if word in emp:
            mm_test[i][j]=1
        j += 1 
    i+=1 


i=0
for x in wl2_test:
    test[x]=mm_test[:,i]
    i+=1



print "\nfinding common feature set..."
testfeats = set(test.columns)
trainfeats=set(train.columns)
bothfeats = testfeats.intersection(trainfeats)
features = list(bothfeats)
test = test[features]
train = train[features]
print "\n.. the features are\n"
print features

print "\n... now for the xgb\n"

N = 200
BOOST = 2000
STOPPING = 20

train.fillna(-1,inplace=True)
test.fillna(-1, inplace=True)



train['target']=target



params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.025,
          "eval_metric" : "rmse",
          "max_depth": 6,
          "subsample": 0.85,
          "colsample_bytree": 0.8,
          "min_child_weight": 5,
          "silent": 1,
          "thread": 1,
          "seed": 52
          }
num_boost_round = BOOST

X_train, X_valid = train_test_split(train, test_size=0.12, random_state=15)
y_train = X_train.target
y_valid = X_valid.target

X_train.drop('target', axis=1, inplace=True)
X_valid.drop('target', axis=1, inplace=True)
dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_valid, y_valid)


watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=STOPPING, verbose_eval=True)
  

print "\nnow for predictions"

print("Make predictions on the test set")
dtest = xgb.DMatrix(test)
test_predictions = gbm.predict(dtest)
result = pd.DataFrame({"ID": test_id, 'five_years': test_predictions})
result.to_csv('out.csv')

quit()




