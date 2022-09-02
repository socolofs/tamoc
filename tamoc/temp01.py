from tamoc import ambient, bent_plume_model


profile = ambient.Profile('../input/my_profile.nc')


bpm = bent_plume_model.Model(profile)

...

bpm.simulate()

bpm.save_txt('../output/state_space_dump_for_sim_00', '../input/', 'my_profile.nc')
bpm.save_sim('../output/sim_00', '../input/', 'my_profile.nc')
