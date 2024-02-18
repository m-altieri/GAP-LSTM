import pandas as pd
import sys

machine = sys.argv[1] # argv[1]: recas / au
assert machine == 'recas' or machine == 'au' or machine == 'recas-argparse'

runs_df = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else 'runs.csv')

for row in runs_df.values:
	run_id, model, dataset, conf, mode, wsize = row

	runname = '{}-{}-{}-{}'.format(model, dataset, conf, mode)
	if wsize > 0:
		runname += '-{}'.format(wsize)

	if machine == 'recas':
		filename = 'jsons/job-' + runname + '.json'
		name = 'altieri-' + runname
		command = 'python3 /lustrehome/altieri/research/src/AT_GCLSTM-dev.py -c {} -o {} -m {} {} {}'.format(conf, 'test-' + runname, mode, model, dataset)
		if wsize > 0:
			command += ' {}'.format(wsize)

		with open(filename, 'w', newline='') as outf:
			outf.write(r'''{
        "name": "%s",
        "command": "%s",
        "shell": true,
        "retries": 99,
        "description": "",
        "cpus": 1,
        "disk": 192000,
        "mem": 128000,
        "gpus": 1,
        "environmentVariables": [],
        "arguments": [],
        "runAsUser": "altieri",
        "owner": "altieri",
        "ownerName": "altieri",
        "container": {
        "type": "mesos",
        "image": "registry-clustergpu.recas.ba.infn.it/altieri/tf-gpu-image:1.0",
        "volumes": [{"containerPath": "/lustrehome/altieri", "hostPath": "/lustrehome/altieri", "mode": "RW"}]
        },
        "schedule": "R1//P1Y",
        "constraints": [["hostname", "LIKE", "hpc-gpu-1-[2,4]-[1-2].recas.ba.infn.it"]]
}''' % (name, command))

		run_all_filename = 'run_all.sh'
		with open(run_all_filename, 'a', newline='') as f:
			command = './submit-to-chronos.sh {}\n'.format(filename)
			f.write(command)
			f.write('sleep 60\n')
	elif machine == 'au':
		filename = 'run_all.bat'
		with open(filename, 'a', newline='') as outf:
			command = 'python AT_GCLSTM.py {} {} {} {} {}'.format(model, dataset, conf, 'test-' + runname, mode)
			if wsize > 0:
				command += ' {}'.format(wsize)
			outf.write('{}\n'.format(command))
	else:
		assert false