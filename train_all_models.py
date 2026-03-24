import os, cv2, librosa, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Extensions
IMAGE_EXTENSIONS={'.jpg','.jpeg','.png'}
AUDIO_EXTENSIONS={'.wav'}

def is_image(f): return os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
def is_audio(f): return os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS

def extract_video_clips(path, clip_len=16, sample_rate=2):
    cap=cv2.VideoCapture(path); frames=[]
    ok=True
    while ok:
        ok,fr=cap.read()
        if ok: frames.append(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames)<clip_len*sample_rate: return []
    clips=[]
    for start in range(0,len(frames)-clip_len*sample_rate+1,clip_len*sample_rate):
        clips.append([frames[start+i*sample_rate] for i in range(clip_len)])
    return clips

def audio_to_spec(path,n_mels=128):
    y,sr=librosa.load(path,sr=None)
    spec=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=n_mels)
    return librosa.power_to_db(spec,ref=np.max)

# =========================
# Dataset
# =========================
class MultiModalDataset(Dataset):
    def __init__(self,img_dir,vid_dir,aud_dir,transform,clip_len=16,sample_rate=2):
        self.data=[]
        self.transform=transform
        self.clip_len=clip_len; self.sample_rate=sample_rate
        for f in os.listdir(img_dir):
            p=os.path.join(img_dir,f)
            if is_image(f): self.data.append(("image",Image.open(p).convert("RGB"),0))
        for f in os.listdir(vid_dir):
            p=os.path.join(vid_dir,f)
            if p.lower().endswith(('.mp4','.avi')):
                for clip in extract_video_clips(p,clip_len,sample_rate):
                    self.data.append(("video",[Image.fromarray(fr) for fr in clip],1))
        for f in os.listdir(aud_dir):
            p=os.path.join(aud_dir,f)
            if is_audio(f):
                self.data.append(("audio",audio_to_spec(p),2))
    def __len__(self): return len(self.data)
    def __getitem__(self,idx):
        m,d,l=self.data[idx]
        if m=="image":
            return {"type":"image","data":self.transform(d)},torch.tensor(l)
        if m=="video":
            frames=[self.transform(fr) for fr in d]
            clip=torch.stack(frames,dim=0) # (T,C,H,W)
            return {"type":"video","data":clip},torch.tensor(l)
        if m=="audio":
            t=torch.tensor(d).unsqueeze(0)
            t=nn.functional.interpolate(t.unsqueeze(0),(128,128)).squeeze(0)
            return {"type":"audio","data":t.float()},torch.tensor(l)

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# Models
# =========================
class AudioCNN(nn.Module):
    def __init__(self,num_classes=3):
        super().__init__()
        m=resnet18(weights=None)
        m.conv1=nn.Conv2d(1,64,7,2,3,bias=False)
        m.fc=nn.Linear(m.fc.in_features,num_classes)
        self.model=m
    def forward(self,x): return self.model(x)

class VideoRNN(nn.Module):
    def __init__(self,hidden_size=256,num_classes=3):
        super().__init__()
        base=resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn=nn.Sequential(*list(base.children())[:-1])
        self.feature_dim=base.fc.in_features
        self.rnn=nn.LSTM(self.feature_dim,hidden_size,1,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)
    def forward(self,x): # x:(B,T,C,H,W)
        B,T,C,H,W=x.shape
        x=x.view(B*T,C,H,W)
        feats=self.cnn(x).view(B,T,-1)
        out,_=self.rnn(feats)
        return self.fc(out[:,-1,:])

# =========================
# Custom collate_fn
# =========================
def custom_collate(batch):
    # Just return the list without stacking (since shapes differ)
    return batch

# =========================
# Train
# =========================
def train():
    img_dir=r"C:\Users\User\OneDrive\Pictures"
    vid_dir=r"C:\Users\User\Videos"
    aud_dir=r"C:\Users\User\Music"
    ds=MultiModalDataset(img_dir,vid_dir,aud_dir,transform)
    loader=DataLoader(ds,batch_size=1,shuffle=True,collate_fn=custom_collate)
    image_model=resnet18(weights=ResNet18_Weights.DEFAULT); image_model.fc=nn.Linear(image_model.fc.in_features,3)
    video_model=VideoRNN(num_classes=3)
    audio_model=AudioCNN(num_classes=3)
    image_model.to(device); video_model.to(device); audio_model.to(device)
    opt=optim.Adam(list(image_model.parameters())+list(video_model.parameters())+list(audio_model.parameters()),lr=5e-4)
    crit=nn.CrossEntropyLoss()
    for epoch in range(5):
        total=0
        for batch in loader:
            inp,lab=batch[0]  # because batch_size=1
            typ=inp["type"]; x=inp["data"].to(device); y=lab.to(device).unsqueeze(0)
            opt.zero_grad()
            if typ=="image": out=image_model(x.unsqueeze(0))
            elif typ=="video": out=video_model(x.unsqueeze(0))
            elif typ=="audio": out=audio_model(x.unsqueeze(0))
            else: continue
            loss=crit(out,y); loss.backward(); opt.step()
            total+=loss.item()
        print(f"Epoch {epoch+1}, Loss {total/len(loader):.4f}")
    torch.save({"image_model":image_model.state_dict(),
                "video_model":video_model.state_dict(),
                "audio_model":audio_model.state_dict()},
               "stego_multi_modal.pth")

if __name__=="__main__":
    train()




