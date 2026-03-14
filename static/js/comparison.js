(function () {
  const data = window.EIPART_DATA;
  if (!data) {
    console.error("Missing assets/manifest.js (window.EIPART_DATA)");
    return;
  }

  if (!window.THREE || !THREE.GLTFLoader || !THREE.OrbitControls) {
    console.error("Three.js, GLTFLoader, or OrbitControls is missing.");
    return;
  }

  const oursHint = document.querySelector("#oursMissing");
  const baselineHint = document.querySelector("#methodMissing");
  const baselineSelection = document.querySelector("#comparisonBaselineSelection");
  const panel = document.querySelector("#comparisonSelectionPanel");
  const selectedIdTag = document.querySelector("#selectedSampleId");
  const explodeSlider = document.querySelector("#explodeRatioSlider");
  const explodeValue = document.querySelector("#explodeRatioValue");
  const oursContainer = document.querySelector("#oursThreeView");
  const baselineContainer = document.querySelector("#baselineThreeView");

  if (!baselineSelection || !panel || !selectedIdTag || !oursContainer || !baselineContainer) {
    console.error("Comparison page elements are missing.");
    return;
  }

  const imageFiles = (data.inputImageFiles || []).filter((name) => !name.startsWith("."));
  const imageMap = new Map(imageFiles.map((name) => [name.replace(/\.[^/.]+$/, ""), name]));
  const sampleIds = Array.from(imageMap.keys()).sort((a, b) => Number(a) - Number(b));
  const methodFiles = data.methodFiles || {};
  const oursMethod = data.oursMethod || "EI-Part";
  const loader = new THREE.GLTFLoader();

  let currentId = sampleIds.includes("007") ? "007" : sampleIds[0];
  let explodeAmount = explodeSlider ? Number(explodeSlider.value) : 0;
  let syncLock = false;
  let activePanel = "ours";

  function viewerPath(method, fileName) {
    return `${data.methodsDir}/${method}/${fileName}`;
  }

  function resolveFileName(method, sampleId) {
    const files = (methodFiles[method] || []).filter((name) => name.toLowerCase().endsWith(".glb"));
    const exact = `${sampleId}.glb`;
    if (files.includes(exact)) {
      return exact;
    }

    const prefixMatches = files.filter((name) => name.startsWith(`${sampleId}-`) || name.startsWith(`${sampleId}(`));
    if (prefixMatches.length > 0) {
      return prefixMatches.sort()[0];
    }

    return null;
  }

  function baselineMissingText(method) {
    if (method === "BANG") {
      return "This sample is not available for Rodin-3D. You can test it on the Rodin-3D website.";
    }
    return "";
  }

  function displayMethodName(method) {
    if (method === "BANG") {
      return "Rodin-3D";
    }
    return method;
  }

  function getSelectedMethod() {
    const option = baselineSelection.selectedOptions && baselineSelection.selectedOptions[0];
    if (option && option.dataset.method) {
      return option.dataset.method;
    }
    return baselineSelection.value;
  }

  function setHint(el, text) {
    el.textContent = text || "";
    el.style.display = text ? "block" : "none";
  }

  function createPanelState(container) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf4f5f7);
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    camera.position.set(0, 0, 2.85);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(container.clientWidth || 100, container.clientHeight || 100);
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.physicallyCorrectLights = true;
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.18;
    controls.rotateSpeed = 0.9;
    controls.zoomSpeed = 0.9;
    controls.enablePan = false;

    const ambient = new THREE.AmbientLight(0xffffff, 1.55);
    const hemisphere = new THREE.HemisphereLight(0xffffff, 0xe7ebf2, 0.8);
    const key = new THREE.DirectionalLight(0xffffff, 1.65);
    const fill = new THREE.DirectionalLight(0xffffff, 1.05);
    const rim = new THREE.DirectionalLight(0xffffff, 0.7);
    key.position.set(7, 10, 9);
    fill.position.set(-8, 5, 7);
    rim.position.set(0, -4, -8);
    scene.add(ambient, hemisphere, key, fill, rim);

    return {
      container,
      scene,
      camera,
      renderer,
      controls,
      model: null,
      parts: [],
      src: null,
      visible: false
    };
  }

  const oursState = createPanelState(oursContainer);
  const baselineState = createPanelState(baselineContainer);

  function resizePanel(panelState) {
    const width = Math.max(panelState.container.clientWidth, 1);
    const height = Math.max(panelState.container.clientHeight, 1);
    panelState.camera.aspect = width / height;
    panelState.camera.updateProjectionMatrix();
    panelState.renderer.setSize(width, height);
  }

  function renderPanel() {
    panel.innerHTML = "";
    sampleIds.forEach((id) => {
      const img = document.createElement("img");
      img.className = "selectable-image";
      img.dataset.id = id;
      img.alt = `input ${id}`;
      img.src = `${data.inputDir}/${imageMap.get(id)}`;
      panel.appendChild(img);
    });
  }

  function renderMethodOptions() {
    baselineSelection.innerHTML = "";
    (data.methods || []).forEach((method) => {
      const option = document.createElement("option");
      option.value = displayMethodName(method);
      option.dataset.method = method;
      option.textContent = displayMethodName(method);
      baselineSelection.appendChild(option);
    });
  }

  function clearModel(panelState) {
    if (panelState.model) {
      panelState.scene.remove(panelState.model);
    }
    panelState.model = null;
    panelState.parts = [];
    panelState.src = null;
    panelState.visible = false;
    panelState.renderer.domElement.style.opacity = "0";
  }

  function fitModelToView(panelState, model, scaleFactor) {
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDimension = Math.max(size.x, size.y, size.z) || 1;
    const targetSize = 1.38;
    const scale = (targetSize / maxDimension) * scaleFactor;

    model.scale.setScalar(scale);
    model.position.sub(center.multiplyScalar(scale));

    panelState.camera.position.set(0, 0, 2.95);
    panelState.controls.target.set(0, 0, 0);
    panelState.controls.update();
  }

  function collectPartNodes(model) {
    const root = model.children && model.children.length === 1 ? model.children[0] : model;
    return (root.children || [])
      .filter((node) => node && node.position && typeof node.position.clone === "function")
      .map((node, index) => {
        const bbox = new THREE.Box3().setFromObject(node);
        const center = bbox.getCenter(new THREE.Vector3());
        let direction = center.clone().normalize();
        if (!isFinite(direction.x) || direction.lengthSq() < 1e-8) {
          direction = new THREE.Vector3(index % 2 === 0 ? 1 : -1, ((index % 3) - 1) * 0.5, 0).normalize();
        }
        return {
          node,
          originalPosition: node.position.clone(),
          direction
        };
      });
  }

  function applyExplode(panelState, ratio) {
    if (!panelState.model || panelState.parts.length === 0) {
      return;
    }

    const distance = ratio * 0.95;
    panelState.parts.forEach((part) => {
      part.node.position.copy(part.originalPosition);
      part.node.position.addScaledVector(part.direction, distance);
    });
  }

  function syncPanels(sourceState, targetState) {
    if (syncLock || !sourceState.visible || !targetState.visible) {
      return;
    }

    syncLock = true;
    targetState.camera.position.copy(sourceState.camera.position);
    targetState.camera.quaternion.copy(sourceState.camera.quaternion);
    targetState.controls.target.copy(sourceState.controls.target);
    targetState.camera.zoom = sourceState.camera.zoom;
    targetState.camera.updateProjectionMatrix();
    targetState.controls.update();
    syncLock = false;
  }

  function loadIntoPanel(panelState, src, hintEl, hintText, scaleFactor) {
    if (!src) {
      clearModel(panelState);
      setHint(hintEl, hintText);
      return;
    }

    setHint(hintEl, "");
    loader.load(
      src,
      (gltf) => {
        clearModel(panelState);
        panelState.model = gltf.scene;
        panelState.model.traverse((node) => {
          if (!node.isMesh || !node.material) {
            return;
          }
          node.castShadow = false;
          node.receiveShadow = false;
          if ("metalness" in node.material) {
            node.material.metalness = 0.02;
          }
          if ("roughness" in node.material) {
            node.material.roughness = Math.min(node.material.roughness ?? 0.9, 0.92);
          }
          node.material.envMapIntensity = 0.55;
          node.material.needsUpdate = true;
        });
        panelState.scene.add(panelState.model);
        panelState.src = src;
        panelState.visible = true;
        panelState.renderer.domElement.style.opacity = "1";
        fitModelToView(panelState, panelState.model, scaleFactor);
        panelState.parts = collectPartNodes(panelState.model);
        applyExplode(panelState, explodeAmount);
      },
      undefined,
      () => {
        clearModel(panelState);
        setHint(hintEl, hintText);
      }
    );
  }

  function updateSelection(sampleId) {
    currentId = sampleId;
    selectedIdTag.textContent = sampleId;

    panel.querySelectorAll(".selectable-image").forEach((img) => {
      img.classList.toggle("selected", img.dataset.id === sampleId);
    });

    const baselineMethod = getSelectedMethod();
    const oursFile = resolveFileName(oursMethod, sampleId);
    const baselineFile = resolveFileName(baselineMethod, sampleId);

    loadIntoPanel(
      oursState,
      oursFile ? viewerPath(oursMethod, oursFile) : null,
      oursHint,
      "Missing EI-Part result.",
      0.9
    );

    loadIntoPanel(
      baselineState,
      baselineFile ? viewerPath(baselineMethod, baselineFile) : null,
      baselineHint,
      baselineMissingText(baselineMethod),
      0.9
    );
  }

  function animate() {
    requestAnimationFrame(animate);
    oursState.controls.update();
    baselineState.controls.update();
    oursState.renderer.render(oursState.scene, oursState.camera);
    baselineState.renderer.render(baselineState.scene, baselineState.camera);
  }

  function bindPanelSync(panelState, panelName) {
    panelState.controls.addEventListener("start", () => {
      activePanel = panelName;
    });
    panelState.controls.addEventListener("change", () => {
      if (activePanel !== panelName) {
        return;
      }
      const target = panelName === "ours" ? baselineState : oursState;
      syncPanels(panelState, target);
    });
  }

  bindPanelSync(oursState, "ours");
  bindPanelSync(baselineState, "baseline");

  renderPanel();
  renderMethodOptions();

  if ((data.methods || []).includes("BANG")) {
    baselineSelection.value = "Rodin-3D";
  }

  updateSelection(currentId);
  resizePanel(oursState);
  resizePanel(baselineState);
  animate();

  window.addEventListener("resize", () => {
    resizePanel(oursState);
    resizePanel(baselineState);
  });

  panel.addEventListener("click", (event) => {
    const img = event.target.closest(".selectable-image");
    if (!img) {
      return;
    }
    updateSelection(img.dataset.id);
  });

  baselineSelection.addEventListener("change", () => {
    updateSelection(currentId);
  });

  if (explodeSlider) {
    explodeSlider.addEventListener("input", (event) => {
      explodeAmount = Number(event.target.value);
      if (explodeValue) {
        explodeValue.textContent = explodeAmount.toFixed(2);
      }
      applyExplode(oursState, explodeAmount);
      applyExplode(baselineState, explodeAmount);
    });
  }

  if (explodeValue) {
    explodeValue.textContent = explodeAmount.toFixed(2);
  }
})();
