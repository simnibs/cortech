import numpy as np
import pyvista as pv
from trame.widgets.vuetify3 import VSelect, VSlider


class InteractivePlotter:
    def __init__(self, fssub):
        self.fssub = fssub


        self.plotter = pv.Plotter(
            notebook=True, off_screen=True
        )
        self.plotter.theme.jupyter_backend= "client"

        kw_brain = dict(
            scalars="curv_bin", cmap="gray", clim=(0, 1), show_scalar_bar=False
        )
        self.plotter.add_composite(self.fssub.brain, **kw_brain)
        self.actor, self.mapper = self.plotter.add_composite(self.fssub.overlays)
        self.item_list = list(self.fssub.overlays["lh"].point_data.keys())
        self.item_list.append("Off")

        self.widget = self.plotter.show(jupyter_kwargs=dict(add_menu_items=self.custom_tools), return_viewer=True)
        state, ctrl = self.widget.viewer.server.state, self.widget.viewer.server.controller
        self.state = state
        self.ctrl = ctrl
        self.ctrl.view_update = self.widget.viewer.update

        @state.change('scalars')
        def set_scalars(scalars=self.fssub.overlays['lh'].active_scalars_name, **kwargs) -> None:
            if scalars in 'Off':
                self.actor.visibility=False
                self.plotter.scalar_bar.visibility=False
            else:
                self.actor.visibility=True
                self.plotter.scalar_bar.title=scalars
                self.plotter.scalar_bar.visibility=True
                self.fssub.overlays.set_active_scalars(scalars)
                data_range_lh = self.fssub.overlays['lh'].get_data_range()
                data_range_rh = self.fssub.overlays['rh'].get_data_range()
                self.mapper.scalar_range = (np.minimum(data_range_lh[0], data_range_rh[0]), np.maximum(data_range_lh[1], data_range_rh[1]))

            self.ctrl.view_update()

        @state.change("opacity")
        def set_opacity(opacity=1.0, **kwargs) -> None:
            self.actor.GetProperty().SetOpacity(opacity)
            self.ctrl.view_update()

    def custom_tools(self):
        VSelect(
                label="Scalars",
                v_model=("scalars", self.fssub.overlays['lh'].active_scalars_name),
                items=("array_list", self.item_list),
                hide_details=True,
                density="compact",
                outlined=True,
                classes="pt-1 ml-2",
                style="max-width: 250px",
            )
        VSlider(
            v_model=("opacity", 1.0),
            min=0.0,
            max=1.0,
            step=0.05,
            dense=True,
            hide_details=True,
            label="Opacity",
            style='width: 300px',
            classes='my-0 py-0 ml-1 mr-1',
        )

    def get_widget(self):
        return self.widget
