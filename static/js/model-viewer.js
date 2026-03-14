// This code is used to add additional functionality to our model-viewers.
document.querySelectorAll('model-viewer.model-viewer-texture').forEach(
    function(modelViewer) {
        // Switches between textured and untextured mode
        modelViewer.setTextured = function(isTextured) { 
            modelViewer.model.materials[0].pbrMetallicRoughness.setMetallicFactor(0.1);
            modelViewer.model.materials[0].pbrMetallicRoughness.setRoughnessFactor(0.8);
            modelViewer.model.materials[0].pbrMetallicRoughness.setBaseColorFactor([0.15, 0.21, 0.3]);
            modelViewer.model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(null);
        };

        // Reset view
        modelViewer.resetView = function () {
            modelViewer.cameraTarget = '0m 0m 0m';
            modelViewer.fieldOfView = '45deg';
            modelViewer.cameraOrbit = '0deg 90deg 2.5m';
            modelViewer.resetTurntableRotation(0);
            modelViewer.jumpCameraToGoal();
        };

        modelViewer.addEventListener('load', async function() {
			// normalize model
			const bbox_size = modelViewer.getDimensions()
			const old_scale = Number(modelViewer.scale.split(" ")[0]);
			const bbox_max = Math.max(bbox_size.x, Math.max(bbox_size.y, bbox_size.z));
			const uniform_scale = old_scale / bbox_max
			
			modelViewer.scale = `${uniform_scale} ${uniform_scale} ${uniform_scale}`;
			
			//modelViewer.currentTexture = await modelViewer.createTexture(modelViewer.texturePath);
            modelViewer.setTextured();
        });
    }
);

