"""Post-outcome seed-151 audit of the original Gate 9 linear subproblem.

This experimenter-only diagnostic does not guide learner selection. It requires
requirements-examples.txt and prints Dykstra/SLSQP linear and actual violations.
The analytic Jacobians below are only for the measured linear subproblem.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from gate9_selected_replay import make_pool,run_method,select_indices
from gate8_retrieval_access import CueRecognizer,view,margins
from behavioral_guard import project_response_bounds
seed=151
images,labels=load_digits(return_X_y=True);images=images/16.
a,test=train_test_split(np.arange(len(labels)),test_size=.2,stratify=labels,random_state=seed)
train,cal=train_test_split(a,test_size=.25,stratify=labels[a],random_state=seed+1)
old,new=train_test_split(cal,test_size=.5,stratify=labels[cal],random_state=seed+2);new=new[:64]
with threadpool_limits(limits=1):
 model=MLPClassifier(hidden_layer_sizes=(24,),activation='tanh',solver='lbfgs',max_iter=200,random_state=seed).fit(np.concatenate([view(images[train],v) for v in ('full','upper','lower')]),np.tile(labels[train],3))
 world=CueRecognizer(model,seed);pool,_=make_pool(world,images,labels,old)
 captured=[]
 def capture(theta):
  captured.append(theta.copy());return {}
 r=run_method(world,pool,.55*images[new]+.22,labels[new],capture,'fixed_mixed',seed,model_corrections=0)
 theta=captured[-1]
 selected=select_indices('fixed_mixed',pool,np.random.default_rng(1))
 def bank(t):
  oldv=[margins(world.probabilities(t,pool[i]['image'][None,:]),[pool[i]['label']])[0] for i in selected]
  p=world.probabilities(t,.55*images[new]+.22)
  return np.r_[oldv,np.mean(np.log(p[np.arange(len(new)),labels[new]]+1e-12))]
 current=bank(theta);target=r['initial_objective']+.05
 h=1e-4;eye=np.eye(24)*h
 rows=np.array([(bank(theta+v)-bank(theta-v))/(2*h) for v in eye]).T
 low=np.r_[[pool[i]['bound'] for i in selected],target]-current
 upper=np.r_[np.full(20,np.inf),target]-current
 g=low[-1]*rows[-1]/np.dot(rows[-1],rows[-1]);g*=min(1.,.5/np.linalg.norm(g))
 def audit(z):
  vals=bank(theta+z)
  return dict(norm=float(np.linalg.norm(z)),linear_violation=float(max(0.,np.max(low-rows@z),np.max(rows@z-upper))),new_progress=float(vals[-1]-r['initial_objective']),actual_old_violation=float(max(0.,np.max(np.array([pool[i]['bound'] for i in selected])-vals[:-1]))))
 z=project_response_bounds(g,rows,low,upper,.5)
 print('current',r['actual_progress'],'Dykstra',audit(z))
 constraints=[{'type':'ineq','fun':lambda z: rows[:-1]@z-low[:-1],'jac':lambda z:rows[:-1]}, {'type':'eq','fun':lambda z:rows[-1]@z-low[-1],'jac':lambda z:rows[-1]}, {'type':'ineq','fun':lambda z:.25-z@z,'jac':lambda z:-2*z}]
 solved=minimize(lambda z:.5*np.dot(z-g,z-g),np.zeros(24),jac=lambda z:z-g,constraints=constraints,method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
 print('SLSQP',solved.success,solved.message,audit(solved.x))
 for fraction in (1.,.5,.25,.125): print('scaled',fraction,audit(solved.x*fraction))
