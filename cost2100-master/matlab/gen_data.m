%% 314513025
%% gen_data.m 基於 demo_model.m 修改，產生5組不同的資料集

% 預設配置
Network = 'Indoor_CloselySpacedUser_2_6GHz';
Link = 'Multiple';
Antenna = 'MIMO_Cyl_patch';
Band = 'Wideband';

% 把產生出來的放/data裡
mkdir('data');

% 產生5 個不同的資料集
for i = 1:5
    fprintf('gen data %d\n', i);
    
    scenario = 'LOS';
    freq =[2.58e9 2.62e9]; 
    snapNum = 1000; % 將100改1000夠多的數來做 train/val/test
    snapRate = 50; 

    % 原本的 closely-spaced users 分佈
    base_MSPos = [   -2.5600    1.7300    2.2300;...
                     -3.0800    1.7300    2.2300;...
                     -2.5600    2.6200    2.5800;...
                     -4.6400    1.7300    2.2300;...
                     -2.5600    4.4000    3.3000;...
                     -3.0800    3.5100    2.9400;...
                     -3.6000    4.4000    3.3000;...
                     -4.1200    4.4000    3.3000;...
                     -4.1200    2.6200    2.5800]; % [x, y, z] (m)
                 
    %創造 5 種不同偏移分佈
    dx = (i - 3) * 5; % -10, -5, 0, 5, 10
    dy = (i - 3) * 3; % -6, -3, 0, 3, 6
    % 改z好像會報錯==
    MSPos = base_MSPos + repmat([dx, dy, 0], size(base_MSPos,1), 1);
    
    MSVelo = repmat([-.25,0,0], 9, 1);
    BSPosCenter = [0.30 -4.37 3.20]; 
    BSPosCenter = BSPosCenter - mean(MSPos);
    MSPos = MSPos - repmat(mean(MSPos),size(MSPos,1),1);
    BSPosSpacing =[0 0 0];
    BSPosNum = 1;

    % cost2100
    [paraEx, paraSt, link, env] = cost2100(Network, scenario, freq, snapRate, snapNum, BSPosCenter, BSPosSpacing, BSPosNum, MSPos, MSVelo);

    %合併天線響應
    BSantEADF = load('BS_Cyl_EADF.mat','F');
    MSantPattern = load('MS_AntPattern_User.mat');
    delta_f = (freq(2)-freq(1))/256; %將頻寬切為 256 個子載波

    % 取得 Impulse Response 與 Transfer Function
    ir_Cyl_Patch = create_IR_Cyl_EADF(link, freq, delta_f, BSantEADF.F, MSantPattern);
    H_transfer = fft(ir_Cyl_Patch, [], 2); % [snapshot, freq(257), ms(9), bs_ant(128)]
    
    % 轉換為Angular-Delay Domain,截取 32x32 的空間-頻率，維度簡化取第1 個 MS的資料
    H_user1 = squeeze(H_transfer(:, :, 1, :)); % size:[snapshot, 257, 128]
    
    % 取 32 個天線，與 125 個頻率子載波
    H_freq_sub = H_user1(:, 1:125, 1:32); % 取前 125 個 subcarrier，前 32 根天線
    H_freq_sub = permute(H_freq_sub,[1, 3, 2]); % size: [snapshot, 32, 125]
    
    % 轉換至 Delay Domain (0-pad到257 後IFFT，再截斷 32)
    H_padded = cat(3, H_freq_sub, zeros(snapNum, 32, 257-125));
    H_delay = ifft(H_padded,[], 3);
    H_delay = H_delay(:, :, 1:32); % size: [snapshot, 32, 32] (Angular-Delay)
    
    % 拆分實虛部跟正規至0~1
    H_delay = H_delay / max(abs(H_delay(:))); 
    H_real = real(H_delay) + 0.5;
    H_imag = imag(H_delay) + 0.5;
    
    HT = zeros(snapNum, 2, 32, 32);
    HT(:, 1, :, :) = H_real;
    HT(:, 2, :, :) = H_imag;
    HF_all = H_freq_sub; % 存頻域供測correlation

    % Train 0.6, Val 0.2, Test 0.2
    train_idx = 1:floor(0.6*snapNum);
    val_idx = floor(0.6*snapNum)+1:floor(0.8*snapNum);
    test_idx = floor(0.8*snapNum)+1:snapNum;
    HT_train = HT(train_idx,:,:,:); HT_val = HT(val_idx,:,:,:); HT_test = HT(test_idx,:,:,:);
    HF_test = HF_all(test_idx,:,:);
    
    save(sprintf('data/DATA_Htrain_ds%d.mat', i), 'HT', '-v7.3');
    % 分開存
    HT = HT_train; save(sprintf('data/DATA_Htrainin_ds%d.mat', i), 'HT');
    HT = HT_val; save(sprintf('data/DATA_Hvalin_ds%d.mat', i), 'HT');
    HT = HT_test; save(sprintf('data/DATA_Htestin_ds%d.mat', i), 'HT');
    
    HF_all = HF_test; save(sprintf('data/DATA_HtestFin_all_ds%d.mat', i), 'HF_all');
    
    fprintf('data %d saved.\n', i);
end