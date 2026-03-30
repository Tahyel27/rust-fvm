use std::{sync::{Arc, Condvar, Mutex, atomic::AtomicBool}, thread};
use std::sync::atomic::Ordering;
use std::sync::mpsc;

use crate::mpscsingle;

#[derive(Default)]
pub struct SimulationContext<Params> {
    step: usize,
    params: Params
}

impl<Params> SimulationContext<Params> {
    fn increment_step(&mut self) { self.step = self.step + 1; }

    pub fn get_step(&self) -> usize {self.step}

    pub fn get_params(&self) -> &Params {&self.params}
}

pub trait SimulationData: Send {
    type SimRes: Send;

    type SimParams: Clone + Send;

    fn update(&mut self, ctx: &SimulationContext<Self::SimParams>) -> ();

    fn send_result(&self, ctx: &SimulationContext<Self::SimParams>) -> Self::SimRes;
}

#[derive(Default)]
pub struct SimulationHandler<T: SimulationData + 'static> {
    params: Arc<Mutex<T::SimParams>>,
    shared_data_ref: Arc<Mutex<Option<T>>>,
    rx: Option<mpscsingle::Receiver<T::SimRes>>,
    tx: Option<mpsc::Sender<Box<dyn FnOnce(&mut T) + Send>>>,
    new_data_flag: Arc<AtomicBool>,
    paused_flag: Arc<(AtomicBool, Condvar, Mutex<()>)>,
    send_freq: usize   
}

impl<SimData: SimulationData + 'static> SimulationHandler<SimData> {
    pub fn new(initial_data: SimData, parameters: SimData::SimParams) -> Self {
        let shared_data_ref = Arc::new(Mutex::new(Some(initial_data)));
        let new_data_flag = Arc::new(AtomicBool::new(false));
        let paused_flag = Arc::new((AtomicBool::new(false), Condvar::new(), Mutex::new(())));
        let params = Arc::new(Mutex::new(parameters));
        Self { params, shared_data_ref, rx: None, tx: None, new_data_flag, paused_flag, send_freq: 1 }
    }

    pub fn send_frequency(mut self, frequency: usize) -> Self {
        self.send_freq = frequency;
        self
    }

    pub fn run(&mut self) {

        let data_opt = {self.shared_data_ref.lock().unwrap().take()};

        if let Some(mut data) = data_opt {

            let send_freq = self.send_freq;
            let new_data_flag = Arc::clone(&self.new_data_flag);
            let shared_data_ref = Arc::clone(&self.shared_data_ref);
            let sim_params = Arc::clone(&self.params);
            let paused_flag = Arc::clone(&self.paused_flag);

            let (tx, rx) = mpscsingle::channel();
            let (ftx, frx) = mpsc::channel();

            self.rx = Some(rx);
            self.tx = Some(ftx);

            thread::spawn(move || {
                let params = { sim_params.lock().unwrap().clone() };

                let mut ctx = SimulationContext{ step: 0, params};
                
                loop {
                    ctx.params = sim_params.lock().unwrap().clone();

                    data.update(&ctx);

                    if ctx.step % send_freq == 0 {
                        let res = data.send_result(&ctx);

                        if tx.send(res).is_err() {
                            break ;
                        }
                    }

                    if new_data_flag.load(std::sync::atomic::Ordering::Relaxed) {
                        let new_data = shared_data_ref.lock().unwrap().take();
                        if let Some(ndata) = new_data {
                            data = ndata;
                        }

                        new_data_flag.store(false, std::sync::atomic::Ordering::Relaxed);
                    }

                    if paused_flag.0.load(std::sync::atomic::Ordering::Relaxed) {
                        let mut guard = paused_flag.2.lock().unwrap();
                        while paused_flag.0.load(std::sync::atomic::Ordering::Relaxed) {
                            guard = paused_flag.1.wait(guard).unwrap();
                        }
                    }

                    if let Ok(f) = frx.try_recv() {
                        f(&mut data);
                    }

                    ctx.increment_step();
                }

            });
        }
    }

    pub fn try_receive(&self) -> Option<SimData::SimRes> {
        match &self.rx {
            Some(rx) => rx.try_recv(),
            None => None
        }
    }

    pub fn modify_data<F>(&self, f: F)
    where
        F: FnOnce(&mut SimData) + Send + 'static
    {
        if let Some(tx) = &self.tx {
            tx.send(Box::new(f)).unwrap();
        }
    }

    pub fn pause(&mut self) {
        self.paused_flag.0.store(true, Ordering::Relaxed);
    }

    pub fn resume(&mut self) {
        self.paused_flag.0.store(false, Ordering::Relaxed);
        self.paused_flag.1.notify_all();
    }

    pub fn get_params(&self) -> SimData::SimParams {
        self.params.lock().unwrap().clone()
    }

    pub fn update_params<F>(&mut self, f: F) where
        F: FnOnce(&mut SimData::SimParams) -> () 
    {
        let mut params = self.params.lock().unwrap();
        f(&mut params)
    }
    

    pub fn set_params(&mut self, params: SimData::SimParams) {
        let mut shared = self.params.lock().unwrap();
        *shared = params;
    }

    pub fn set_data(&mut self, data: SimData) {
        let mut shared = self.shared_data_ref.lock().unwrap();
        let _ = shared.insert(data);
        self.new_data_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}