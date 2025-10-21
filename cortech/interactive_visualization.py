import numpy as np
import pyvista as pv
from trame.widgets.vuetify3 import VSelect, VSlider
import matplotlib as mpl
from cortech import freesurfer

class InteractivePlotter:
    def __init__(self, fssub, name=None):
        self.fssub = fssub
        self.name = name or f"vis_{id(self)}"

        self.plotter = pv.Plotter(notebook=True, off_screen=True)
        self.plotter.theme.jupyter_backend = "client"

        self.annotations = freesurfer.ANNOT

        kw_brain = dict(
            scalars="curv_bin", cmap="gray", clim=(0, 1), show_scalar_bar=False
        )
        self.plotter.add_composite(self.fssub.brain, **kw_brain)
        self.actor, self.mapper = self.plotter.add_composite(self.fssub.overlays)
        self.item_list = list(self.fssub.overlays["lh"].point_data.keys())
        self.item_list.append("Off")

        self.widget = self.plotter.show(
            jupyter_kwargs=dict(add_menu_items=self.custom_tools), return_viewer=True
        )
        self.state, self.ctrl = (
            self.widget.viewer.server.state,
            self.widget.viewer.server.controller,
        )
        # self.state = state
        # self.ctrl = ctrl
        self.ctrl.view_update = self.widget.viewer.update

        @self.state.change(f"{self.name}_scalars")
        def set_scalars(**kwargs):
            scalars = kwargs.get(f"{self.name}_scalars")
            self._set_scalars(scalars)

        @self.state.change(f"{self.name}_opacity")
        def set_opacity(**kwargs):
            opacity = kwargs.get(f"{self.name}_opacity")
            self._set_opacity(opacity)

    def _set_opacity(self, opacity=1.0) -> None:
        self.actor.GetProperty().SetOpacity(opacity)
        self.ctrl.view_update()

    def _set_scalars(self, scalars) -> None:
        if scalars in "Off":
            self.actor.visibility = False
            self.plotter.scalar_bar.visibility = False
        else:
            self.actor.visibility = True
            self.plotter.scalar_bar.title = scalars
            self.fssub.overlays.set_active_scalars(scalars)
            data_tmp = np.hstack((self.fssub.overlays['lh'].active_scalars, self.fssub.overlays['rh'].active_scalars))
            if scalars in self.annotations:
                self.mapper.scalar_range = (
                    np.min(data_tmp),
                    np.max(data_tmp),
                )
                print(np.unique(self.fssub.overlays['lh'][scalars]))
                self.mapper.lookup_table.cmap = self.fssub.clut[scalars]
            else:
                data_tmp = np.hstack((self.fssub.overlays['lh'].active_scalars, self.fssub.overlays['rh'].active_scalars))
                self.mapper.scalar_range = (
                    np.percentile(data_tmp, 5),
                    np.percentile(data_tmp, 95),
                )
                self.mapper.lookup_table.cmap = mpl.colormaps['viridis']

        self.plotter.scalar_bar.visibility = True
        self.ctrl.view_update()

    def custom_tools(self):
        prefix = self.name
        VSelect(
            label="Scalars",
            v_model=(
                f"{prefix}_scalars",
                self.fssub.overlays["lh"].active_scalars_name,
            ),
            items=("array_list", self.item_list),
            hide_details=True,
            density="compact",
            outlined=True,
            classes="pt-1 ml-2",
            style="max-width: 250px",
        )
        VSlider(
            v_model=(f"{prefix}_opacity", 1.0),
            min=0.0,
            max=1.0,
            step=0.05,
            dense=True,
            hide_details=True,
            label="Opacity",
            style="width: 300px",
            classes="my-0 py-0 ml-1 mr-1",
        )

    def get_widget(self):
        return self.widget
