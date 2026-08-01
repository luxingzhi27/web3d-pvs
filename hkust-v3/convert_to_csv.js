const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, 'assets', 'trajectories');

if (fs.existsSync(dir)) {
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.json'));
    
    files.forEach(file => {
        const filePath = path.join(dir, file);
        const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        
        let trajectoryArray = [];
        if (data && data.trajectory) {
            trajectoryArray = data.trajectory;
        } else if (Array.isArray(data)) {
            trajectoryArray = data;
        }
        
        if (trajectoryArray.length > 0) {
            let csv = 'time,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,target_x,target_y,target_z\n';
            trajectoryArray.forEach(frame => {
                csv += `${frame.time},${frame.position.x},${frame.position.y},${frame.position.z},${frame.rotation.x},${frame.rotation.y},${frame.rotation.z},${frame.target.x},${frame.target.y},${frame.target.z}\n`;
            });
            
            const newFile = filePath.replace('.json', '.csv');
            fs.writeFileSync(newFile, csv, 'utf8');
            console.log(`Converted ${file} to ${path.basename(newFile)}`);
            fs.unlinkSync(filePath); // remove old json
        }
    });
} else {
    console.log("Trajectories folder not found:", dir);
}
