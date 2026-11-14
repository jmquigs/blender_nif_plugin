mkdir -p io_scene_nif/dependencies
ln -s "$PWD/pyffi/pyffi" io_scene_nif/dependencies/pyffi

mkdir -p ~/.config/blender/2.79/scripts/addons
ln -s "$PWD/io_scene_nif" ~/.config/blender/2.79/scripts/addons/io_scene_nif